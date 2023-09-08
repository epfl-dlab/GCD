# cheatsheet

### number of beam

`model.hparams_overrides.inference.hf_generation_params.num_beams=$n_beam`

### num_shots

`data_module.num_shots=$num_shots`

### specify top n data at runtime

`datamodule.top_k=100`
`datamodule.debug=True datamodule.debug_k=100`

### build few-shot prompts

```terminal
python build_few_shot_prompt.py prompter=el_aida_basic_2.yaml
```


### Specify pgf at runtime

```terminal
model.gf_constraint_module.pgf="$pgf"
```

### Run inference with gf-based constraints

```terminal
MODEL=llama_sc_beam4
DATAMODULE=sdg_text_davinci_003_pc_small_fs2_sc
/mnt/u14157_ic_nlp_001_files_nfs/dlabdata1/llama_hf/7B

python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer=fsdp \
model.pretrained_model_name_or_path=/mnt/u14157_ic_nlp_001_files_nfs/dlabdata1/llama_hf/7B \
 +model.use_gf=true
```


### Download grammars

- genie_llama_fully_expanded_rebel_medium.pgf: `gdown 1adgkcfBrIZLcnpU8vwTbcRX_F0yjP5zT`
- genie_llama_subject_collapsed_rebel_medium.pgf: `gdown 1th2Plsb4Nxb-NapSdF9tLyVwxWefUjnd`

### print hydra config

```terminal
MODEL=llama_sc_beam4
DATAMODULE=sdg_text_davinci_003_pc_small_fs2_sc_100
LENGTH_PENALTY=1.0
python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE model.pretrained_model_name_or_path=/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B trainer=fsdp trainer.devices=2   model.hparams_overrides.inference.hf_generation_params.length_penalty=$LENGTH_PENALTY model.hparams_overrides.inference.idx_of_return_sequence_as_output=1 --cfg job --resolve
```

### Evaluate llama

```terminal
WANDB_PATH=smirnov-space/SynthIE/k3j0bbx3
python run_process_predictions.py +experiment/process_predictions=complete_rebel wandb_run_path=$WANDB_PATH
```

```terminal
MODEL=llama_sc_beam4
DATAMODULE=sdg_text_davinci_003_pc_small_fs2_sc_100
LENGTH_PENALTY=3.0
python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE model.pretrained_model_name_or_path=/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B trainer=fsdp trainer.devices=2   model.inference.hf_generation_params.length_penalty=$LENGTH_PENALTY model.inference.idx_of_return_sequence_as_output=0
```


TODO : add runai config to docker image ; copy llama weight to disk to speed up first loading

- No Constrain
`++model.constraint_module=null /`

- Model Parallelism
`trainer=fsdp`

### Select GPU device

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer.devices=\[1\]`

### batch size

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE datamodule.batch_size=4`


### Overwrite pgf

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE model.gf_constraint_module.pgf=EL_OTF_llama_aida_test`

### provide llama path

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer=cpu model.pretrained_model_name_or_path=/dlabdata1/llama_hf/13B`

### Build `llama_tokenizable`

`python -m scripts.non_problematic_constrained_world --tokenizer_full_name /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B --tokenizer_short_name llama --constrained_world_id genie`

### T5

[T5 training](https://huggingface.co/docs/transformers/model_doc/t5?highlight=t5#training)
- model._shift_right:https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model.forward.example

### Check preprocessing input

Preprocessing is handled by `generic_collator.py`, an intermedia between `tokenizer` and `model`.

Below line in `Genie_lan_t5.py`
```python
        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)
```




### Data for few-shot learning

- "The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State Administration for Market Regulation and is a government agency under the parent organization, the State Council of the People's Republic of China. Its headquarters is located in Haidian District, China."
- "Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter."
- "Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language."
- "The NHL Stadium Series is a sport that consists of ice hockey."
- "Abhishek Pictures is a film production company based in Hyderabad."


- "[s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] State_Administration_for_Market_Regulation [r] instance of [o] Government_agency [r] parent organization [o] State_Council_of_the_People's_Republic_of_China [r] country [o] China [r] headquarters location [o] Haidian_District [e]",
- "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e]"
- "[s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [r] developer [o] Swedish_National_Heritage_Board [r] product or material produced [o] Resource_Description_Framework [r] file format [o] XML [r] file format [o] JSON [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]"
- "[s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e]"
- "[s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]"

prompt = """The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State Administration for Market Regulation and is a government agency under the parent organization, the State Council of the People's Republic of China. Its headquarters is located in Haidian District, China. -> [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] State_Administration_for_Market_Regulation [r] instance of [o] Government_agency [r] parent organization [o] State_Council_of_the_People's_Republic_of_China [r] country [o] China [r] headquarters location [o] Haidian_District [e];

Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter. -> [s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e];

Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language. -> [s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [r] developer [o] Swedish_National_Heritage_Board [r] product or material produced [o] Resource_Description_Framework [r] file format [o] XML [r] file format [o] JSON [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e];

The NHL Stadium Series is a sport that consists of ice hockey. -> [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e];

Abhishek Pictures is a film production company based in Hyderabad. -> [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e];

The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus. -> """


short_prompt = """
The NHL Stadium Series is a sport that consists of ice hockey. -> [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e];

Abhishek Pictures is a film production company based in Hyderabad. -> [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e];

The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus. -> """


new_prompt="""
- Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter.
- The NHL Stadium Series is a sport that consists of ice hockey.
- Abhishek Pictures is a film production company based in Hyderabad.
- The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus.

- [s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e]
- [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e]
- [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]
- """

```python
from src.models import IEHFModelPL

model_7b = IEHFModelPL(from_pretrained=True, pretrained_model_name_or_path=
"/dlabdata1/llama_hf/7B",
                       linearization_class_id="subject_collapsed", collator_parameters=
                       {"max_input_length": 2048, "padding": "longest", "truncation": True},
                       inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                           "early_stopping": False,
                                                           "encoder_no_repeat_ngram_size": 0
                           , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0,
                                                           "max_new_tokens": 256}}
                       )
texts = [prompt]

override_models_default_hf_generation_parameters = {
    "num_beams": 10,
    "num_return_sequences": 1,
    "return_dict_in_generate": True,
    "output_scores": True,
    "seed": 123,
    "length_penalty": 0.8
}

output = model_7b.sample(texts,
                         return_generation_outputs=True,
                         convert_to_triplets=True,
                         **override_models_default_hf_generation_parameters)
print(model_7b.hparams.pretrained_model_name_or_path.split("/")[-1])
print(model_7b.tokenizer.batch_decode(output['generation_outputs'].sequences, skip_special_tokens=True))
print(output['grouped_decoded_outputs'][0])
```

### Llama+ constrained decoding

# state_id="et_id", this is strange. it should be

```python
"sub_id": ["rel_id"],
"rel_id": ["obj_id"],
"obj_id": ["rel_id", "et_id"],
"et_id": ["sub_id"],
```

### llama hparams

```python

model_7b.hparams["collator_parameters"]["max_input_length"]=1024
"""
Out[5]:
"collator_parameters":   {'max_input_length': 24, 'padding': 'longest', 'truncation': True}
"from_pretrained":               True
"hf_config":                     LlamaConfig {
  "_name_or_path": "/home/saibo/Research/llama-7B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.29.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}

"inference":                     {'hf_generation_params': {'num_beams': 1, 'num_return_sequences': 1, 'early_stopping': False, 'encoder_no_repeat_ngram_size': 0, 'no_repeat_ngram_size': 0, 'temperature': 1.0, 'length_penalty': 1.0, 'max_new_tokens': 24}}
"linearization_class_id":        fully_expanded
"pretrained_model_name_or_path": /home/saibo/Research/llama-7B
"""
```

### Why so slow on RUNAI ?

```
%prun     model = load_llama(model=args.model, model_dir=args.model_dir,max_input_length=args.max_input_length, max_new_tokens=
   ...: args.max_new_tokens)
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      323  891.898    2.761  891.977    2.762 serialization.py:994(load_tensor)
     2031   13.271    0.007   13.271    0.007 {method 'to' of 'torch._C._TensorBase' objects}
      368    6.964    0.019    6.966    0.019 {built-in method tensor}
```


### Genie Data Format

```json
{
  "id": 44904,
  "entities": [
    {
      "surfaceform": "Family_Matters",
      "uri": "Q593838"
    },
    {
      "surfaceform": "1989",
      "uri": "Q2425"
    },
    {
      "surfaceform": "List_of_Family_Matters_characters",
      "uri": "Q6570699"
    }
  ],
  "relations": [
    {
      "surfaceform": "set in period",
      "uri": "P2408"
    },
    {
      "surfaceform": "list of characters",
      "uri": "P1881"
    }
  ],
  "triplets": [
    {
      "subject": {
        "surfaceform": "Family_Matters",
        "uri": "Q593838"
      },
      "object": {
        "surfaceform": "1989",
        "uri": "Q2425"
      },
      "predicate": {
        "surfaceform": "set in period",
        "uri": "P2408"
      }
    },
    {
      "subject": {
        "surfaceform": "Family_Matters",
        "uri": "Q593838"
      },
      "object": {
        "surfaceform": "List_of_Family_Matters_characters",
        "uri": "Q6570699"
      },
      "predicate": {
        "surfaceform": "list of characters",
        "uri": "P1881"
      }
    }
  ],
  "text": "Family Matters is set in the year 1989 and features a list of characters, known as the List of Family Matters characters."
}

```


### Format for Entity Sense Disambiguiation

```json
{
  "id": 0,
  "input": "Eu rejects [START_ENT] German [END_ENT] call to boycott British lamb Peter Blackburn Brussels 1996 08 22 The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep Germany s representative to the European Union s veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer We do n t support any such recommendation because we do n t see any grounds for it the Commission s chief spokesman Nikolaus van der Pas told a news",
  "output": [
    {
      "answer": "Germany",
      "provenance": [
        {
          "title": "Germany"
        }
      ]
    }
  ],
  "meta": {
    "left_context": "Eu rejects",
    "right_context": "call to boycott British lamb Peter Blackburn Brussels 1996 08 22 The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep Germany s representative to the European Union s veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer We do n t support any such recommendation because we do n t see any grounds for it the Commission s chief spokesman Nikolaus van der Pas told a news",
    "mention": "German"
  },
  "candidates": [
    "German language",
    "Germany",
    "Germans",
    "Nazi Germany",
    "German American",
    "German Empire",
    "Ethnic Germans",
    "Wehrmacht",
    "Kriegsmarine",
    "Canadians of German ethnicity",
    "German Brazilian",
    "Patriarch German of Serbia",
    "Cinema of Germany",
    "German Township, Vanderburgh County, Indiana",
    "German colonial empire",
    "Media Control Charts",
    "German Wikipedia",
    "Germanic peoples",
    "West Germany",
    "Germany national football team",
    "Holy Roman Empire",
    "German (mythology)",
    "German Australian",
    "Weimar Republic",
    "Imperial German Navy",
    "Bundeswehr",
    "German Argentine",
    "Obadiah German",
    "German name",
    "German Confederation",
    "Germans in Omaha, Nebraska",
    "Luftwaffe",
    "German phonology",
    "Germania",
    "Standard German",
    "German (parish)",
    "German Township, Harrison County, Ohio",
    "German Chilean",
    "Nazism",
    "German alphabet",
    "German Township, Auglaize County, Ohio",
    "German Township, Marshall County, Indiana",
    "German Township, Clark County, Ohio",
    "Bundesliga",
    "German Township, Richland County, Illinois",
    "German Peruvian",
    "1983 German Grand Prix",
    "German Township, St. Joseph County, Indiana",
    "German, New York",
    "East Germany",
    "German Township, Bartholomew County, Indiana",
    "German-American internment",
    "German Township, Fulton County, Ohio",
    "Military Administration in France (Nazi Germany)",
    "Antonio German",
    "Germans in Czechoslovakia (1918–1938)",
    "1966 German Grand Prix",
    "2002 German Grand Prix",
    "Walser German",
    "German rock",
    "Germany in the Eurovision Song Contest 2009",
    "Jim German",
    "German Namibians",
    "Axis powers",
    "Trusty John",
    "Wer wird Millionär? (Germany)",
    "Edward German",
    "BMW 3 Series (E46)",
    "1951 German Grand Prix",
    "German Tatarinov",
    "German Beriyev",
    "Germany men's national ice hockey team",
    "German Aerospace Center",
    "Prussia",
    "Singapore",
    "The Six Swans",
    "Deutsche Bahn",
    "The Three Dogs",
    "German East Africa",
    "Germany at the 2010 Winter Olympics",
    "Danube Swabians",
    "German Colony, Jerusalem",
    "English language",
    "German Apukhtin",
    "Emergency medical services in Germany",
    "German art",
    "German Garrido",
    "German Pinelli",
    "Condor Legion",
    "Bible translations into German",
    "1962 German Grand Prix",
    "Germans of Hungary",
    "German Goldenshteyn",
    "Carpathian Germans",
    "German Felk",
    "German dialects",
    "Swiss German",
    "Cannock Chase German war cemetery",
    "Afrika Korps"
  ]
}

```


Subject: Request for Testing Node Upgrade to Accommodate Cutting-Edge Large Models

Dear IC-IT Team,

I hope this email finds you well.
The topic of this email is regarding the testing node gpu010.rcp.epfl.ch from the new cluster.
Some of my students have started to use the testing node gpu010.rcp.epfl.ch to run experiments.
After conducting experiments and investigations, we have encountered an unfortunate OOM issue while attempting to run the 65B model on the testing node gpu010.rcp.epfl.ch, which is equipped with 8 A100 GPUs (80GB).
We believe that this issue is due to the fact that the testing node gpu010.rcp.epfl.ch is not equipped with enough memory to accommodate the 65B model.
We think this issue is representative and will become more prevalent when the cluster is deployed in production, so we would like to report this issue and request an upgrade to the testing node to test it again.

To provide some context, the llama-65B model we are working with contains 65B parameters and is 122GB in size(stored in fp16).

To utilize the PyTorch fsdp (fully sharded data parallel) mode(which is a mainstream distributed training mode),  we require each process (one per GPU) to copy the entire model during the initialization phase for sharding purposes.

With our current setup of 4 GPUs(we only made use of 4 out of 8), we would require 122GB x 4 = 488GB solely for the model during initialization.

Regrettably, this surpasses the available memory(504GB) of the gpu010.rcp.epfl.ch testing node, leading to out-of-memory (OOM) errors and eventual kernel termination.

Indeed, we think that the configuration of the testing node may be suboptimal for experimentation with large models.

Intuitively, we would expect that the testing node would be equipped with at least the same amount of memory as the GPU memory(640GB) to be able to initialize the model and transfer each shard to the GPU.

So the RAM should be larger than the total amount of GPU memory.

We also checked the configuration of the cloud service providers and here is what we found:
- AWS p4d.24xlarge: 8 A100 GPUs (80GB) + 1152 GB RAM
- AWS p4d.12xlarge: 8 A100 GPUs (40GB) + 1152 GB RAM
- Azure NC96ads A100 v4: 4 A100 GPUs (80GB) + 880 GB RAM
- Azure NC96asr A100 v4: 8 A100 GPUs (80GB) + 1900 GB RAM
- GCP a2-highgpu-8g: 8 A100 GPUs (40GB) + 680 GB RAM

Given these circumstances, I would like to propose that for future cluster nodes, we should consider the following configurations:
- 8 A100 GPUs (80GB) + at least 800 GB RAM, ideally 1000 GB RAM
- 4 A100 GPUs (80GB) + 500 GB RAM

We understand that this might involve additional costs and logistical considerations, but we believe it would be a worthwhile investment to enhance the research capabilities for the IC community.

If you think this is a reasonable request, please let us know and we can rerun the experiments on the upgraded testing node to verify the results.

Thank you for your time and consideration.
