import random
import copy
import json
import pathlib

from transformers import LlamaTokenizer

import video_instruct_dataset
from video_llama.processors.video_processor import load_video
from video_llama.processors import AlproVideoTrainProcessor

class BDD_Video_Dataset(video_instruct_dataset.Video_Instruct_Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root,num_video_query_token=32,tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/',data_type = 'video', model_type='vicuna'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
            
        with data_path.open(encoding='utf-8') as f:
            self.annotations = [json.loads(line) for line in f]

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([video_instruct_dataset.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[video_instruct_dataset.DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type
    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                video_path = self._get_video_path(sample)
                conversation_list = [sample['QA']]

                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling ="uniform", return_msg = True
                )
                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                
                sources = video_instruct_dataset.preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=self.num_video_query_token,msg = msg)
                new_sources = video_instruct_dataset.convert_source_vicuna_format(sources)
                
                if self.model_type =='vicuna':
                    data_dict = video_instruct_dataset.preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = video_instruct_dataset.preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
        }