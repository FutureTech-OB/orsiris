"""Optional local HF integration. Random tiny weights; no network or model download."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import torch
from repsteer.backends import HFBackend
from repsteer.interventions import Intervention, InterventionSession
from repsteer.metrics import choice_probabilities


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Install .[hf] for HF integration test")
class HFIntegrationTests(unittest.TestCase):
    def test_local_qwen35_hybrid_image_text_adapter(self):
        import transformers
        if not hasattr(transformers,"Qwen3_5ForConditionalGeneration"):
            self.skipTest("Installed transformers does not support Qwen3.5")
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer
        with tempfile.TemporaryDirectory() as td:
            raw=Tokenizer(BPE(unk_token="[UNK]"))
            raw.pre_tokenizer=ByteLevel(add_prefix_space=False)
            raw.decoder=ByteLevelDecoder()
            raw.train_from_iterator(["A B fair unfair The outcome is unfair Accept Reject"],
                BpeTrainer(vocab_size=280,special_tokens=["[UNK]","[PAD]","[EOS]"],initial_alphabet=ByteLevel.alphabet()))
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",pad_token="[PAD]",eos_token="[EOS]")
            tokenizer.chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %} Answer:"
            tokenizer.save_pretrained(td)
            vocab=raw.get_vocab_size()
            config=Qwen3_5Config(text_config={"vocab_size":vocab+4,"hidden_size":32,"intermediate_size":48,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,
                "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_num_key_heads":2,
                "linear_num_value_heads":2,"layer_types":["linear_attention","full_attention"],
                "max_position_embeddings":512,"pad_token_id":1,"eos_token_id":2},
                vision_config={"depth":1,"hidden_size":16,"intermediate_size":24,"num_heads":2,
                "out_hidden_size":32,"num_position_embeddings":16,"patch_size":2,"temporal_patch_size":2},
                image_token_id=vocab,video_token_id=vocab+1,vision_start_token_id=vocab+2,vision_end_token_id=vocab+3)
            model=Qwen3_5ForConditionalGeneration(config)
            self.assertLess(sum(p.numel() for p in model.parameters()),100_000)
            model.save_pretrained(td)
            backend=HFBackend({"path":td,"model_id":"tiny-qwen","chat_template":True,"enable_thinking":False,
                               "auto_class":"image_text","dtype":"float32","device_map":"cpu"})
            case={"event":"The outcome is unfair.","choices":[{"id":"a","text":"Accept","value":1},{"id":"b","text":"Reject","value":0}]}
            encoded=backend.encode(case,[0,1])
            with torch.inference_mode(): baseline=backend.forward(encoded.ids)
            with InterventionSession(backend.layers,[Intervention(0,"add",torch.ones(32),strength=.1,position=encoded.event_end)]) as s, torch.inference_mode():
                changed=backend.forward(encoded.ids)
            self.assertEqual(len(s.records),1)
            self.assertFalse(torch.equal(baseline,changed))
            with torch.inference_mode(): restored=backend.forward(encoded.ids)
            torch.testing.assert_close(baseline,restored,atol=0,rtol=0)

    def test_local_causal_lm_loading_tokenization_and_hook(self):
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        with tempfile.TemporaryDirectory() as td:
            raw=Tokenizer(WordLevel({"[UNK]":0,"[PAD]":1,"[EOS]":2,"A":3,"B":4,"fair":5,"unfair":6},unk_token="[UNK]"))
            raw.pre_tokenizer=Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",pad_token="[PAD]",eos_token="[EOS]")
            tokenizer.save_pretrained(td)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=7,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=2048))
            model.save_pretrained(td)
            backend=HFBackend({"path":td,"model_id":"tiny-local","chat_template":False,"dtype":"float32","device_map":"cpu"})
            case={"event":"The outcome is unfair.","choices":[{"id":"a","text":"Accept","value":1},{"id":"b","text":"Reject","value":0}]}
            encoded=backend.encode(case,[0,1]); labels=backend.label_ids(2)
            with torch.inference_mode(): baseline=backend.forward(encoded.ids)
            with InterventionSession(backend.layers,[Intervention(0,"add",torch.ones(16),strength=.1,position=encoded.event_end)]) as session, torch.inference_mode():
                steered=backend.forward(encoded.ids)
            self.assertEqual(len(session.records),1)
            self.assertFalse(torch.equal(baseline,steered))
            self.assertAlmostEqual(sum(choice_probabilities(steered,labels)["probabilities"]),1,places=6)
            with torch.inference_mode(): restored=backend.forward(encoded.ids)
            torch.testing.assert_close(restored,baseline,rtol=0,atol=0)
