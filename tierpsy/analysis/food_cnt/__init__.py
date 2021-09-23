from tierpsy.helper.params.models_path import DFLT_MODEL_FOOD_CONTOUR

from .getFoodContour import getFoodContour

def args_(fn, param):
  requirements = ['SKE_INIT']

  argkws = dict(
          mask_file = fn['masked_image'],
          skeletons_file = fn['skeletons'],
          use_nn_food_cnt = param.p_dict['use_nn_food_cnt'],
          model_path = DFLT_MODEL_FOOD_CONTOUR, #i am putting this a separated paramters because by default i will not copy the pretrained model since it is too big
          solidity_th = 0.98,
          _is_debug = False

  )
  #arguments used by AnalysisPoints.py
  return {
        'func': getFoodContour,
        'argkws': argkws,
        'input_files' : [fn['masked_image']],
        'output_files': [fn['skeletons'], fn['masked_image']],
        'requirements' : requirements
    }
