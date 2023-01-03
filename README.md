# Awesome-Parameter-Efficient-Transfer-Learning
A collection of parameter-efficient transfer learning papers focusing on computer vision and multimodal domains.

## Content

- [Why Parameter Efficient?](#why-parameter-efficient)
- [Keywords Convention](#keywords-convention)
- [Papers](#papers)
  - [Prompt](#prompt)
  - [Adapter](#adapter)
  - [Unified](#unified)
  - [Ohters](#others)
- [Contribution](#contribution)
  - [Contributors](#contributors)
  - [Contributing to this paper list](#contributing-to-this-paper-list)
- [Acknowledgement](#acknowledgement)

## Why Parameter Efficient?

Pre-training, then fully fine-tuning is a long standing paradigm in deep learning. However, as pre-trained models are scaling up, *e.g.* GPT-3(175B params), fully fine-tuning them on various downstream tasks has a high risk of overfitting. Moreover, in practice, it would be costly to train and store a large model for each task. To overcome the above issues, researchers started to explore **Parameter-Efficient Transfer Learning** which aims at adapting large-scale pre-trained model to various downstream tasks by modifying as less parameter as possible.Inspired by the great advances in NLP domain and the continuous trend of scaling up models, scholars in computer vision and multimodal domains also join the research craze.

## Keywords Convention

We follow the general idea of [PromptPapers](https://github.com/thunlp/PromptPapers) to label the papers.

![](https://img.shields.io/badge/CoOp-blue) The abbreviation of the work.

![](https://img.shields.io/badge/Image_Classification-green) The main explored task of the work.

![](https://img.shields.io/badge/Other-orange) Other important information of the work.

## Papers

### Prompt

- **Learning to Prompt for Vision-Language Models,** IJCV 2022 (arXiv:2109.01134). ![](https://img.shields.io/badge/CoOp-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Text_Prompt-orange)

  *Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu.* [[Paper](https://arxiv.org/abs/2109.01134)][[Code](https://github.com/KaiyangZhou/CoOp)]

- **Prompting Visual-Language Models for Efficient Video Understanding,** ECCV 2022 (arXiv:2112.04478). ![](https://img.shields.io/badge/Action_Recognition,_Action_Localization,_Text_Video_Retrieval-green) ![](https://img.shields.io/badge/Text_Prompt-orange)

  *Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, Weidi Xie.* [[Paper](https://arxiv.org/abs/2112.04478)][[Code](https://github.com/ju-chen/Efficient-Prompt)]

- **Domain Adaptation via Prompt Learning,** arXiv:	arXiv:2202.06687. ![](https://img.shields.io/badge/DAPL-blue) ![](https://img.shields.io/badge/Domain_Adaption-green) ![](https://img.shields.io/badge/Text_Prompt-orange)

  *Chunjiang Ge, Rui Huang, Mixue Xie, Zihang Lai, Shiji Song, Shuang Li, Gao Huang.* [[Paper](https://arxiv.org/abs/2202.06687)][Code]

- **Conditional Prompt Learning for Vision-Language Models,** CVPR 2022 (arXiv:2203.05557). ![](https://img.shields.io/badge/CoCoOp-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Visual_Conditional_Text_Prompt-orange)

  *Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu.* [[Paper](https://arxiv.org/abs/2203.05557)][[Code](https://github.com/KaiyangZhou/CoOp)]

- **Visual Prompt Tuning,** ECCV 2022 (arXiv:2203.12119). ![](https://img.shields.io/badge/VPT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Token_level_Prompt-orange)

  *Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, Ser-Nam Lim.* [[Paper](https://arxiv.org/abs/2203.12119)][[Code](https://github.com/kmnp/vpt)]

- **Exploring Visual Prompts for Adapting Large-Scale Models,** arXiv:2203.17274. ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, Phillip Isola.* [[Paper](https://arxiv.org/abs/2203.17274)][[Code](https://github.com/hjbahng/visual_prompting)]

- **Class-Aware Visual Prompt Tuning for Vision-Language Pre-Trained Model,** arXiv:2208.08340. ![](https://img.shields.io/badge/CAVPT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Yinghui Xing, Qirui Wu, De Cheng, Shizhou Zhang, Guoqiang Liang, Yanning Zhang.* [[Paper](https://arxiv.org/abs/2208.08340)][Code]

- **MaPLe: Multi-modal Prompt Learning,** arXiv:2210.03117. ![](https://img.shields.io/badge/MaPLe-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, Fahad Shahbaz Khan.* [[Paper](https://arxiv.org/abs/2210.03117)][[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)]

- **Unified Vision and Language Prompt Learning,** arXiv:2210.07225. ![](https://img.shields.io/badge/UPT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, Chen Change Loy.* [[Paper](https://arxiv.org/abs/2210.07225)][[Code](https://github.com/yuhangzang/UPT)]

- **Understanding and Improving Visual Prompting: A Label-Mapping Perspective,** arXiv:2211.11635. ![](https://img.shields.io/badge/ILM_VP-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Aochuan Chen, Yuguang Yao, Pin-Yu Chen, Yihua Zhang, Sijia Liu.* [[Paper](https://arxiv.org/abs/2211.11635)][[Code](https://github.com/OPTML-Group/ILM-VP)]

- **VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval,** arXiv:2211.12764. ![](https://img.shields.io/badge/VoP-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Siteng Huang, Biao Gong, Yulin Pan, Jianwen Jiang, Yiliang Lv, Yuyuan Li, Donglin Wang.* [[Paper](https://arxiv.org/abs/2211.12764)][[Code](https://github.com/bighuang624/VoP)]

- **Unleashing the Power of Visual Prompting At the Pixel Level,** arXiv:2212.10556. ![](https://img.shields.io/badge/EVP-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Junyang Wu, Xianhang Li, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, Cihang Xie.* [[Paper](https://arxiv.org/abs/2212.10556)][[Code](https://github.com/UCSC-VLAA/EVP)]


### Adapter

- **VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks,** CVPR 2022 (arXiv:2112.06825). 

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2112.06825)][[Code](https://github.com/ylsung/VL_adapter)] ![](https://img.shields.io/badge/VL_Adapter-blue) ![](https://img.shields.io/badge/Image_and_Video_QA,_Caption,_Visual_Reasoning-green) ![](https://img.shields.io/badge/MultiTask_Learning-orange)

- **ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning,** NeurIPS 2022 (arXiv:2206.13559). ![](https://img.shields.io/badge/ST_Adapter-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Temporal_Modeling-orange)

  *Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, Hongsheng Li.* [[Paper](https://arxiv.org/abs/2206.13559)][[Code](https://github.com/linziyi96/st-adapter)]

- **Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets,** arXiv:2208.07463. ![](https://img.shields.io/badge/Conv_Adapter-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Design_for_ConvNet-orange)

  *Hao Chen, Ran Tao, Han Zhang, Yidong Wang, Wei Ye, Jindong Wang, Guosheng Hu, Marios Savvides.* [[Paper](https://arxiv.org/abs/2208.07463)][Code]

- **Cross-Modal Adapter for Text-Video Retrieval,** arXiv:2211.09623. ![](https://img.shields.io/badge/CM_Adapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Haojun Jiang, Jianke Zhang, Rui Huang, Chunjiang Ge, Zanlin Ni, Jiwen Lu, Jie Zhou, Shiji Song, Gao Huang.* [[Paper](https://arxiv.org/abs/2211.09623)][[Code](https://github.com/LeapLabTHU/Cross-Modal-Adapter)]
  
- **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**， EMNLP 2022 (arXiv:2210.04284). ![](https://img.shields.io/badge/-SparseAdapter-blue) ![](https://img.shields.io/badge/-GLUE%20Benchmark-green) ![](https://img.shields.io/badge/-Pretrained%20Language%20Model-orange)
 
  *Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao.*[[Paper](https://arxiv.org/pdf/2210.04284.pdf)][[Code](https://github.com/Shwai-He/SparseAdapter)]


### Unified

- **Towards a Unified View of Parameter-Efficient Transfer Learning,** ICLR 2022 (arXiv:2110.04366). ![](https://img.shields.io/badge/Translation,_Summarization,_Language_Understanding,_Text_Classification-green) ![](https://img.shields.io/badge/Unified_View-orange)

  *Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.* [[Paper](https://arxiv.org/abs/2110.04366)][[Code](https://github.com/jxhe/unify-parameter-efficient-tuning)]

- **Neural Prompt Search,** arXiv:2206.04673. ![](https://img.shields.io/badge/NOAH-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Unified_Search_Framework-orange)

  *Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu.* [[Paper](https://arxiv.org/abs/2206.04673)][[Code](https://github.com/ZhangYuanhan-AI/NOAH)]

### Others

- Check out [thunlp/DeltaPapers](https://github.com/thunlp/DeltaPapers) if you are interested in the progress of NLP domain.

- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning,** NeurIPS 2022 (arXiv:2206.06522). ![](https://img.shields.io/badge/LST-blue) ![](https://img.shields.io/badge/GLUE,_VQA,_Visual_Reasoning,_Image_Caption-green) ![](https://img.shields.io/badge/Side_Tuning-orange)

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2206.06522)][[Code](https://github.com/ylsung/Ladder-Side-Tuning)]

## Contribution

### Contributors

<!-- Copy-paste in your Readme.md file -->

<a href="https://github.com/jianghaojun/Awesome-Parameter-Efficient-Transfer-Learning/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jianghaojun/Awesome-Parameter-Efficient-Transfer-Learning" />
</a>

### Contributing to this paper list

- Here is the tutorial of [contributing to others projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
-  First, think about which category the work should belong to.
-  Second, use the same format as the others to describe the work. Note that there should be an empty line between the title and the author's list, and take care of the indentation.
-  Then, add [keywords tags](#keywords-convention). Add the pdf link of the paper. If it is an arxiv publication, we prefer /abs/ format to /pdf/ format.

## Acknowledgement

The structure of this repository is following [thunlp/DeltaPapers](https://github.com/thunlp/DeltaPapers) which focuses on collecting awesome parameter-efficient transfer learning papers in nature language processing domain. Check out their repository if you are interested in the progress of NLP domain.



