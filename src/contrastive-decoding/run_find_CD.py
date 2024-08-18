from chatgpt_find_divergence_prompts import find_diverging_texts
import sys

target = sys.argv[1].lower()
gens_per_prefix = int(sys.argv[2])
       
if target in ['wihp']:
       for i in range(5):
              print("\n\n\n\nATTEMPT " + str(i) + ":")
              print("include_prefix_in_divergences=False, sequential = False,")
              find_diverging_texts(save_texts_loc="find_CD_outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_log.txt", 
                     model_name="NousResearch/Llama-2-7b-hf",
                     starting_model_path="microsoft/Llama2-7b-WhoIsHarryPotter",
                     comparison_model_path="NousResearch/Llama-2-7b-hf",
                     tokenizer_family="NousResearch/Llama-2-7b-hf",
                     starting_model_weight=1,
                     comparison_model_weight=0,
                     generation_length=40, 
                     limit_to_starting_model_top_p=0.95, 
                     generations_per_prefix=gens_per_prefix, 
                     n_cycles_ask_chatgpt=5, 
                     include_prefix_in_divergences=False,
                     sequential = False,
                     quantize=True)
              print("include_prefix_in_divergences=False, sequential = True,")
              find_diverging_texts(save_texts_loc="find_CD_outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_log.txt", 
                     model_name="NousResearch/Llama-2-7b-hf",
                     starting_model_path="microsoft/Llama2-7b-WhoIsHarryPotter",
                     comparison_model_path="NousResearch/Llama-2-7b-hf",
                     tokenizer_family="NousResearch/Llama-2-7b-hf",
                     starting_model_weight=1,
                     comparison_model_weight=0,
                     generation_length=40, 
                     limit_to_starting_model_top_p=0.95, 
                     generations_per_prefix=gens_per_prefix, 
                     n_cycles_ask_chatgpt=5, 
                     include_prefix_in_divergences=False,
                     sequential = True,
                     quantize=True)
              print("include_prefix_in_divergences=True, sequential = False,")
              find_diverging_texts(save_texts_loc="find_CD_outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_log.txt", 
                     model_name="NousResearch/Llama-2-7b-hf",
                     starting_model_path="microsoft/Llama2-7b-WhoIsHarryPotter",
                     comparison_model_path="NousResearch/Llama-2-7b-hf",
                     tokenizer_family="NousResearch/Llama-2-7b-hf",
                     starting_model_weight=1,
                     comparison_model_weight=0,
                     generation_length=40, 
                     limit_to_starting_model_top_p=0.95, 
                     generations_per_prefix=gens_per_prefix, 
                     n_cycles_ask_chatgpt=5, 
                     include_prefix_in_divergences=True,
                     sequential = False,
                     quantize=True)
              print("include_prefix_in_divergences=True, sequential = True,")
              find_diverging_texts(save_texts_loc="find_CD_outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_log.txt", 
                     model_name="NousResearch/Llama-2-7b-hf",
                     starting_model_path="microsoft/Llama2-7b-WhoIsHarryPotter",
                     comparison_model_path="NousResearch/Llama-2-7b-hf",
                     tokenizer_family="NousResearch/Llama-2-7b-hf",
                     starting_model_weight=1,
                     comparison_model_weight=0,
                     generation_length=40, 
                     limit_to_starting_model_top_p=0.95, 
                     generations_per_prefix=gens_per_prefix, 
                     n_cycles_ask_chatgpt=5, 
                     include_prefix_in_divergences=True,
                     sequential = True,
                     quantize=True)