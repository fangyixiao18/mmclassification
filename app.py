# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable

os.system('python -m pip install openmim')
os.system('python -m mim install mmengine')
os.system('python -m mim install mmcv')
os.system('python -m mim install -e ".[multimodal]"')

import gradio as gr
from mmpretrain.apis import ImageClassificationInferencer, get_model
from functools import partial


class InferencerCache:
    max_size = 2
    _cache = []

    @classmethod
    def get_instance(cls, instance_name, callback: Callable):
        if len(cls._cache) > 0:
            for i, cache in enumerate(cls._cache):
                if cache[0] == instance_name:
                    # Re-insert to the head of list.
                    cls._cache.insert(0, cls._cache.pop(i))
                    return cache[1]

        if len(cls._cache) == cls.max_size:
            cls._cache = cls._cache[:cls.max_size - 1]
        instance = callback()
        cls._cache.insert(0, (instance_name, instance))
        return instance


class ImageClassificationTab:

    def __init__(self) -> None:
        self.short_list = [
            'cn-clip_resnet50_zeroshot-cls_cifar100',
            'cn-clip_vit-base-p16_zeroshot-cls_cifar100',
            'cn-clip_vit-large-p14_zeroshot-cls_cifar100',
        ]
        self.tab = self.create_ui()
        self.current_text = None

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_classification_models',
                    elem_classes='select_model',
                    choices=self.short_list,
                    value='cn-clip_resnet50_zeroshot-cls_cifar100',
                )
                in_image = gr.Image(
                    value=None,
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor')
                in_text = gr.Textbox(
                    value='鸟,猫,狗',
                    label='中文标签',
                    info="请输入中文类别, 例如 '鸟,猫,狗'",
                    interactive=True,
                )
                gr.Examples(
                    examples=[
                        os.path.join("examples", e)
                        for e in os.listdir("examples")
                    ],
                    inputs=in_image,
                    outputs=in_image)

            with gr.Column():
                out_cls = gr.Label(
                    label='Result',
                    num_top_classes=5,
                    elem_classes='cls_result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )

                run_button.click(
                    self.inference,
                    inputs=[select_model, in_image, in_text],
                    outputs=out_cls,
                )

    def inference(self, model, image, text):
        clean_text = text.strip(' ').replace('，', ',')
        inferencer_name = self.__class__.__name__ + model + clean_text

        text_classes = clean_text.split(',')

        if self.current_text is not None and self.current_text == clean_text:
            pass
        else:
            model = get_model(
                model, pretrained=True, text_prototype=text_classes)

        inferencer = InferencerCache.get_instance(
            inferencer_name,
            partial(
                ImageClassificationInferencer, model, classes=text_classes))

        result = inferencer(image)[0]['pred_scores'].tolist()

        if inferencer.classes is not None:
            classes = inferencer.classes
        else:
            classes = list(range(len(result)))

        return dict(zip(classes, result))


if __name__ == '__main__':
    title = 'ChineseCLIP Zero-shot Classification '
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(f'# {title}')
        with gr.Tabs():
            with gr.TabItem('Image Classification'):
                ImageClassificationTab()

    demo.launch(server_name='0.0.0.0')
