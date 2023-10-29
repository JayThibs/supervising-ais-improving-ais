# Adapted from: https://github.com/zjunlp/EasyEdit/blob/main/examples/run_zsre_llama2.py


import os.path
import sys
# Expects EasyEdit to be installed in the same directory as supervising-ais-improving-ais
# EasyEdit: https://github.com/zjunlp/EasyEdit.git
sys.path.append('../../EasyEdit')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset
from easyeditor import EditTrainer, MENDTrainingHparams


def direct_edit(
               editing_method,
               hparams_dir,
               data_path,
               ds_size = None,
               metrics_save_dir = './easy_edit_output'
    ):

    if editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    else:
        raise NotImplementedError

    test_data = json.load(open(data_path, 'r', encoding='utf-8'))

    if ds_size is not None:
        test_data = random.sample(test_data, ds_size)

    prompts = [test_data_['src'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
    target_new = [edit_data_['alt'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    subject = [edit_data_['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(hparams_dir)

    #if editing_method == 'IKE':
    #    train_data_path = os.path.join(data_dir, 'zsre_mend_train_10000.json')
    #    train_ds = ZsreDataset(train_data_path)
    #    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    #    encode_ike_facts(sentence_model, train_ds, hparams)
    #else:
    #    train_ds = None
    train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    json.dump(metrics, open(os.path.join(metrics_save_dir, f'{editing_method}_results.json'), 'w'), indent=4)
    return metrics, edited_model



def train_mend_hypernet(hparam_path = './hparams/TRAINING/MEND/gpt2-xl', train_path = '../../data/zsre/zsre_mend_train.json', eval_path = '../../data/zsre/zsre_mend_eval.json'):
    hparams = MENDTrainingHparams.from_hparams(hparam_path)
    train_ds = ZsreDataset(train_path, config=hparams)
    eval_ds = ZsreDataset(eval_path, config=hparams)

    trainer = EditTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    return trainer

def apply_mend(hparam_path, prompts, ground_truth, target_new, locality_inputs):
    hparams = MENDHyperParams.from_hparams(hparam_path)
    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        keep_original_weight=True
    )

    return metrics, edited_model



