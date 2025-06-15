from npuperf.classes.stages.MainInputParserStages import parse_json_workload
from npuperf.classes.stages.SimpleMappingGeneratorStage import MappingGeneratorStage
from npuperf.classes.stages.Stage import MainStage, Stage

from npuperf.inputs.HW.Meta_prototype import accelerator


class Dummy(Stage):

    def is_leaf(self):
        return True

    def run(self):
        yield None, self.kwargs


mapping = MainStage([MappingGeneratorStage, Dummy], accelerator=accelerator).run()[0][1]['json_mapping_path_or_dict']

# nn_list = [
#     # 'mobile_vit',
#     # 'swin_tiny_v1',
#     # 'vit_b_16',
#     # 'bert_small',
#     'diffusion_unet',
#     'diffusion_text_encoder',
#     'diffusion_vae_encoder',
# ]
# nn_list = ['mv2', 'mv1', 'fcn8s', 'fsrcnn2x', 'resnet50', 'inceptionv1']
nn_list = ['bert_small']

if __name__ == '__main__':
    for nn in nn_list:
        print(f'Parsing {nn} ...', end=' ')
        parse_json_workload(json_path_or_NN=f'npuperf/inputs/hhb_networks/{nn}.json',
                            mapping_path_or_dict=mapping,
                            export_wl=True,
                            merge_activation_function=False)
        print(f'Parse Network {nn} DONE!')
