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

- **Pro-tuning: Unified Prompt Tuning for Vision Tasks,** arXiv:2207.14381. ![](https://img.shields.io/badge/Protuning-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Xing Nie, Bolin Ni, Jianlong Chang, Gaomeng Meng, Chunlei Huo, Zhaoxiang Zhang, Shiming Xiang, Qi Tian, Chunhong Pan.* [[Paper](https://arxiv.org/abs/2207.14381)][Code]

- **P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting,** arXiv:2208.02812. ![](https://img.shields.io/badge/P2P-blue) ![](https://img.shields.io/badge/PointCloud-green) ![](https://img.shields.io/badge/2D_to_3D-orange)

  *Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, Jiwen Lu.* [[Paper](https://arxiv.org/abs/2208.02812)][[Code](https://github.com/wangzy22/P2P)]

- **Class-Aware Visual Prompt Tuning for Vision-Language Pre-Trained Model,** arXiv:2208.08340. ![](https://img.shields.io/badge/CAVPT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Yinghui Xing, Qirui Wu, De Cheng, Shizhou Zhang, Guoqiang Liang, Yanning Zhang.* [[Paper](https://arxiv.org/abs/2208.08340)][Code]

- **Prompt Tuning with Soft Context Sharing for Vision-Language Models,** arXiv:2208.13474. ![](https://img.shields.io/badge/SoftCPT-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Kun Ding, Ying Wang, Pengzhang Liu, Qiang Yu, Haojian Zhang, Shiming Xiang, Chunhong Pan.* [[Paper](https://arxiv.org/abs/2208.13474)][Code]

- **Language-Aware Soft Prompting for Vision & Language Foundation Models,** arXiv:2210.01115. ![](https://img.shields.io/badge/LASP-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Language_Aware_Prompt-orange)

  *Adrian Bulat, Georgios Tzimiropoulos.* [[Paper](https://arxiv.org/abs/2210.01115)][Code]

- **Prompt Learning with Optimal Transport for Vision-Language Models,** arXiv:2210.01253. ![](https://img.shields.io/badge/Image_Classification-green)

  *Guangyi Chen, Weiran Yao, Xiangchen Song, Xinyue Li, Yongming Rao, Kun Zhang.* [[Paper](https://arxiv.org/abs/2210.01253)][Code]

- **MaPLe: Multi-modal Prompt Learning,** arXiv:2210.03117. ![](https://img.shields.io/badge/MaPLe-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, Fahad Shahbaz Khan.* [[Paper](https://arxiv.org/abs/2210.03117)][[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)]

- **SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models,** arXiv:2210.03794. ![](https://img.shields.io/badge/SVL_Adapter-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Omiros Pantazis, Gabriel Brostow, Kate Jones, Oisin Mac Aodha.* [[Paper](https://arxiv.org/abs/2210.03794)][[Code](https://github.com/omipan/svl_adapter)]

- **Unified Vision and Language Prompt Learning,** arXiv:2210.07225. ![](https://img.shields.io/badge/UPT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, Chen Change Loy.* [[Paper](https://arxiv.org/abs/2210.07225)][[Code](https://github.com/yuhangzang/UPT)]

- **CPL: Counterfactual Prompt Learning for Vision and Language Models,** arXiv:2210.10362. ![](https://img.shields.io/badge/CPL-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Xuehai He, Diji Yang, Weixi Feng, Tsu-Jui Fu, Arjun Akula, Varun Jampani, Pradyumna Narayana, Sugato Basu, William Yang Wang, Xin Eric Wang.* [[Paper](https://arxiv.org/abs/2210.10362)][[Code](https://github.com/yuhangzang/UPT)]

- **Understanding and Improving Visual Prompting: A Label-Mapping Perspective,** arXiv:2211.11635. ![](https://img.shields.io/badge/ILM_VP-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Aochuan Chen, Yuguang Yao, Pin-Yu Chen, Yihua Zhang, Sijia Liu.* [[Paper](https://arxiv.org/abs/2211.11635)][[Code](https://github.com/OPTML-Group/ILM-VP)]

- **Texts as Images in Prompt Tuning for Multi-Label Image Recognition,** arXiv:2211.12739. ![](https://img.shields.io/badge/TaI_DPT-blue) ![](https://img.shields.io/badge/Multilabel_Image_Classification-green)

  *Zixian Guo, Bowen Dong, Zhilong Ji, Jinfeng Bai, Yiwen Guo, Wangmeng Zuo.* [[Paper](https://arxiv.org/abs/2211.12739)][[Code](https://github.com/guozix/TaI-DPT)]

- **VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval,** arXiv:2211.12764. ![](https://img.shields.io/badge/VoP-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/Joint_Prompt-orange)

  *Siteng Huang, Biao Gong, Yulin Pan, Jianwen Jiang, Yiliang Lv, Yuyuan Li, Donglin Wang.* [[Paper](https://arxiv.org/abs/2211.12764)][[Code](https://github.com/bighuang624/VoP)]

- **Unleashing the Power of Visual Prompting At the Pixel Level,** arXiv:2212.10556. ![](https://img.shields.io/badge/EVP-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Pixel_Level_Prompt-orange)

  *Junyang Wu, Xianhang Li, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, Cihang Xie.* [[Paper](https://arxiv.org/abs/2212.10556)][[Code](https://github.com/UCSC-VLAA/EVP)]

### Adapter

- **VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks,** CVPR 2022 (arXiv:2112.06825). 

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2112.06825)][[Code](https://github.com/ylsung/VL_adapter)] ![](https://img.shields.io/badge/VL_Adapter-blue) ![](https://img.shields.io/badge/Image_and_Video_QA,_Caption,_Visual_Reasoning-green) ![](https://img.shields.io/badge/MultiTask_Learning-orange)

- **AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition,** NeurIPS 2022 (arXiv:2205.13535). ![](https://img.shields.io/badge/AdaptFormer-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Temporal_Modeling-orange)

  *Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo.* [[Paper](https://arxiv.org/abs/2205.13535)][[Code](https://github.com/ShoufaChen/AdaptFormer)]

- **Zero-Shot Video Question Answering via Frozen Bidirectional Language Models,** NeurIPS 2022 (arXiv:2206.08155). ![](https://img.shields.io/badge/FrozenBiLM-blue) ![](https://img.shields.io/badge/VQA-green)

  *Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, Cordelia Schmid.* [[Paper](https://arxiv.org/abs/2206.08155)][[Code](https://github.com/antoyang/FrozenBiLM)]

- **ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning,** NeurIPS 2022 (arXiv:2206.13559). ![](https://img.shields.io/badge/ST_Adapter-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Temporal_Modeling-orange)

  *Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, Hongsheng Li.* [[Paper](https://arxiv.org/abs/2206.13559)][[Code](https://github.com/linziyi96/st-adapter)]

- **Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets,** arXiv:2208.07463. ![](https://img.shields.io/badge/Conv_Adapter-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Design_for_ConvNet-orange)

  *Hao Chen, Ran Tao, Han Zhang, Yidong Wang, Wei Ye, Jindong Wang, Guosheng Hu, Marios Savvides.* [[Paper](https://arxiv.org/abs/2208.07463)][Code]

- **Effective Adaptation in Multi-Task Co-Training for
Unified Autonomous Driving,** NeurIPS 2022 (arXiv:2209.08953). ![](https://img.shields.io/badge/LV_Adapter-blue) ![](https://img.shields.io/badge/Detection,_Segmentation-green)

  *Xiwen Liang, Yangxin Wu, Jianhua Han, Hang Xu, Chunjing Xu, Xiaodan Liang.* [[Paper](https://arxiv.org/abs/2209.08953)][Code]

- **Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks,** NeurIPS 2022 (arXiv:2210.03265). ![](https://img.shields.io/badge/Polyhistor-blue) ![](https://img.shields.io/badge/Dense_Vision_Tasks-green) ![](https://img.shields.io/badge/MultiTask_Learning-orange)

  *Yen-Cheng Liu, Chih-Yao Ma, Junjiao Tian, Zijian He, Zsolt Kira.* [[Paper](https://arxiv.org/abs/2210.03265)][Code]

- **Cross-Modal Adapter for Text-Video Retrieval,** arXiv:2211.09623. ![](https://img.shields.io/badge/CM_Adapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Haojun Jiang, Jianke Zhang, Rui Huang, Chunjiang Ge, Zanlin Ni, Jiwen Lu, Jie Zhou, Shiji Song, Gao Huang.* [[Paper](https://arxiv.org/abs/2211.09623)][[Code](https://github.com/LeapLabTHU/Cross-Modal-Adapter)]

- **Vision Transformers are Parameter-Efficient Audio-Visual Learners,** arXiv:2212.07983. ![](https://img.shields.io/badge/LAVISH-blue) ![](https://img.shields.io/badge/Audio_Visual_Tasks-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Yan-Bo Lin, Yi-Lin Sung, Jie Lei, Mohit Bansal, Gedas Bertasius.* [[Paper](https://arxiv.org/abs/2212.07983)][[Code](https://github.com/GenjiB/LAVISH)]

  Take away message: Pre-trained vision transformer can deal with audio data by representing 1D raw audio signal as 2D audio image.

- **Multimodal Video Adapter for Parameter Efficient Video Text Retrieval,** arXiv:2301.07868. ![](https://img.shields.io/badge/MV-Adapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Bowen Zhang, Xiaojie Jin, Weibo Gong, Kai Xu, Zhao Zhang, Peng Wang, Xiaohui Shen, Jiashi Feng.* [[Paper](https://arxiv.org/abs/2301.07868)][Code]

- **UniAdapter: Unified Parameter-Efficient Transfer Learning for Cross-modal Modeling,** arXiv:2302.06605. ![](https://img.shields.io/badge/UniAdapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval,_Text_Image_Retrieval,_VideoQA,_VQA-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Haoyu Lu, Mingyu Ding, Yuqi Huo, Guoxing Yang, Zhiwu Lu, Masayoshi Tomizuka, Wei Zhan.* [[Paper](https://arxiv.org/abs/2302.06605)][[Code](https://github.com/RERV/UniAdapter)]

- **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**, EMNLP 2022 (arXiv:2210.04284). ![](https://img.shields.io/badge/-SparseAdapter-blue) ![](https://img.shields.io/badge/-GLUE%20Benchmark-green) ![](https://img.shields.io/badge/-Pretrained%20Language%20Model-orange)
 
  *Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao.* [[Paper](https://arxiv.org/pdf/2210.04284.pdf)][[Code](https://github.com/Shwai-He/SparseAdapter)]

### Unified

- **Towards a Unified View of Parameter-Efficient Transfer Learning,** ICLR 2022 (arXiv:2110.04366). ![](https://img.shields.io/badge/Translation,_Summarization,_Language_Understanding,_Text_Classification-green) ![](https://img.shields.io/badge/Unified_View-orange)

  *Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.* [[Paper](https://arxiv.org/abs/2110.04366)][[Code](https://github.com/jxhe/unify-parameter-efficient-tuning)]

- **Neural Prompt Search,** arXiv:2206.04673. ![](https://img.shields.io/badge/NOAH-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Unified_Search_Framework-orange)

  *Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu.* [[Paper](https://arxiv.org/abs/2206.04673)][[Code](https://github.com/ZhangYuanhan-AI/NOAH)]

### Others

- Check out [thunlp/DeltaPapers](https://github.com/thunlp/DeltaPapers) if you are interested in the progress of NLP domain.

- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning,** NeurIPS 2022 (arXiv:2206.06522). ![](https://img.shields.io/badge/LST-blue) ![](https://img.shields.io/badge/GLUE,_VQA,_Visual_Reasoning,_Image_Caption-green) ![](https://img.shields.io/badge/Side_Tuning-orange)

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2206.06522)][[Code](https://github.com/ylsung/Ladder-Side-Tuning)]

- **Convolutional Bypasses Are Better Vision Transformer Adapters,** arXiv:2207.07039. ![](https://img.shields.io/badge/Convpass-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Convolution_Inductive_Bias-orange)

  *Shibo Jie, Zhi-Hong Deng.* [[Paper](https://arxiv.org/abs/2207.07039)][[Code](https://github.com/JieShibo/PETL-ViT)]

- **Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning,** NeurIPS 2022 (arXiv:2210.08823). ![](https://img.shields.io/badge/SSF-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Feature_Scale_&_Shift-orange)

  *Dongze Lian, Daquan Zhou, Jiashi Feng, Xinchao Wang.* [[Paper](https://arxiv.org/abs/2210.08823)][[Code](https://github.com/dongzelian/SSF)]

- **FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer,** AAAI 2023 (arXiv:2212.03145). ![](https://img.shields.io/badge/FacT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Tensor_Decomposition-orange)

  *Shibo Jie, Zhi-Hong Deng.* [[Paper](https://arxiv.org/abs/2212.03145)][[Code](https://github.com/JieShibo/PETL-ViT)]

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



