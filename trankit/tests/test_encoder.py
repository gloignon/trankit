# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
import os
import tempfile
import unittest

from trankit.config import Config
from trankit.encoder import (adapter_reduction_factor, embedding_dim, embedding_key,
                             load_tokenizer, set_embedding)


class EncoderTests(unittest.TestCase):
    def test_official_keys_unchanged(self):
        c = Config()
        set_embedding(c, 'xlm-roberta-base')
        self.assertEqual((c.embedding_name, c.embedding_source), ('xlm-roberta-base',) * 2)
        self.assertEqual(embedding_dim(c), 768)
        set_embedding(c, 'xlm-roberta-large')
        self.assertEqual(embedding_dim(c), 1024)

    def test_custom_key_is_path_safe(self):
        key = embedding_key('almanach/camembert-base')
        self.assertNotIn('/', key)
        self.assertEqual(key, 'almanach--camembert-base')

    def test_reduction_factor_matches_stock(self):
        self.assertEqual(adapter_reduction_factor(768), 6)
        self.assertEqual(adapter_reduction_factor(1024), 4)

    @unittest.skipUnless(os.environ.get('TRANKIT_TEST_SP_DIR'), 'set TRANKIT_TEST_SP_DIR to a local SentencePiece model dir')
    def test_local_encoder_end_to_end(self):
        from transformers import XLMRobertaConfig, XLMRobertaModel
        src = os.environ['TRANKIT_TEST_SP_DIR']
        with tempfile.TemporaryDirectory() as d:
            XLMRobertaModel(XLMRobertaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                             intermediate_size=128)).save_pretrained(d)
            load_tokenizer(_cfg(src)).save_pretrained(d)
            c = Config()
            c._cache_dir = os.path.join(d, 'cache')
            set_embedding(c, d)
            self.assertEqual(embedding_dim(c), 64)
            from trankit.models.base_models import Multilingual_Embedding
            m = Multilingual_Embedding(c)
            self.assertEqual(m.xlmr_dim, 64)
            self.assertTrue(load_tokenizer(c).tokenize('bonjour le monde'))


def _cfg(src):
    c = Config()
    c._cache_dir = src
    set_embedding(c, src)
    return c


if __name__ == '__main__':
    unittest.main()
