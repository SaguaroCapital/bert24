# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# yapf: disable
from bert_layers import (BertAlibiEmbeddings, BertAlibiEncoder, BertForMaskedLM,
                         BertForSequenceClassification, BertResidualGLU,
                         BertAlibiLayer, BertLMPredictionHead, BertModel,
                         BertOnlyMLMHead, BertOnlyNSPHead, BertPooler,
                         BertPredictionHeadTransform, BertSelfOutput,
                         BertAlibiUnpadAttention, BertAlibiUnpadSelfAttention)
from bert_layers import(FlexBertAbsoluteEmbeddings, FlexBertForMaskedLM, 
                        FlexBertForMultipleChoice, FlexBertForSequenceClassification, 
                        FlexBertForTokenClassification, FlexBertModel, 
                        FlexBertPaddedAttention, FlexBertPaddedPostNormLayer,
                        FlexBertPaddedPreNormLayer, FlexBertSansPositionEmbeddings,
                        FlexBertUnpadAttention, FlexBertUnpadPostNormLayer,
                        FlexBertUnpadPreNormLayer)
# yapf: enable
from bert_padding import (
    IndexFirstAxis,
    IndexPutFirstAxis,
    index_first_axis,
    index_put_first_axis,
    pad_input,
    unpad_input,
    unpad_input_only,
)
from src.bert_layers.configuration_bert import BertConfig
from src.bert_layers.configuration_bert import FlexBertConfig

from hf_bert import create_hf_bert_classification, create_hf_bert_mlm
from mosaic_bert import create_mosaic_bert_classification, create_mosaic_bert_mlm
from flex_bert import create_flex_bert_classification, create_flex_bert_mlm
__all__ = [
    "FlexBertAbsoluteEmbeddings",    
    "FlexBertForMaskedLM",    
    "FlexBertForMultipleChoice",    
    "FlexBertForSequenceClassification",    
    "FlexBertForTokenClassification",    
    "FlexBertModel",    
    "FlexBertPaddedAttention",    
    "FlexBertPaddedPostNormLayer",    
    "FlexBertPaddedPreNormLayer",    
    "FlexBertSansPositionEmbeddings",    
    "FlexBertUnpadAttention",    
    "FlexBertUnpadPostNormLayer",    
    "FlexBertUnpadPreNormLayer",
    "BertAlibiEmbeddings",
    "BertAlibiEncoder",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "BertResidualGLU",
    "BertAlibiLayer",
    "BertLMPredictionHead",
    "BertModel",
    "BertOnlyMLMHead",
    "BertOnlyNSPHead",
    "BertPooler",
    "BertPredictionHeadTransform",
    "BertSelfOutput",
    "BertAlibiUnpadAttention",
    "BertAlibiUnpadSelfAttention",
    "BertConfig",
    "IndexFirstAxis",
    "IndexPutFirstAxis",
    "index_first_axis",
    "index_put_first_axis",
    "pad_input",
    "unpad_input",
    "unpad_input_only",
    "create_hf_bert_classification",
    "create_hf_bert_mlm",
    "create_mosaic_bert_classification",
    "create_mosaic_bert_mlm",
    "create_flex_bert_classification",
    "create_flex_bert_mlm",
    "FlexBertConfig",
    # These are commented out because they only exist if CUDA is available
    # 'flash_attn_func_bert',
    # 'flash_attn_qkvpacked_func_bert'
]
