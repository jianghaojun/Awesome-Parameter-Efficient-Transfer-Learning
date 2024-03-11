# Awesome-Parameter-Efficient-Transfer-Learning
A collection of parameter-efficient transfer learning papers focusing on computer vision and multimodal domains.

## Content

- [Why Parameter Efficient?](#why-parameter-efficient)
- [Keywords Convention](#keywords-convention)
- [Papers](#papers)
  - [Prompt](#prompt)
  - [Adapter](#adapter)
  - [Unified](#unified)
  - [Others](#others)
- [Contribution](#contribution)
  - [Contributors](#contributors)
  - [Contributing to this paper list](#contributing-to-this-paper-list)
- [Acknowledgement](#acknowledgement)

## Why Parameter Efficient?

Pre-training, then fully fine-tuning is a long standing paradigm in deep learning. However, as pre-trained models are scaling up, *e.g.* GPT-3(175B params), fully fine-tuning them on various downstream tasks has a high risk of overfitting. Moreover, in practice, it would be costly to train and store a large model for each task. To overcome the above issues, researchers started to explore **Parameter-Efficient Transfer Learning** which aims at adapting large-scale pre-trained model to various downstream tasks by modifying as less parameter as possible. Inspired by the great advances in NLP domain and the continuous trend of scaling up models, scholars in computer vision and multimodal domains also join the research craze.

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

- **Self-Supervised Convolutional Visual Prompts,** arXiv:2303.00198. ![](https://img.shields.io/badge/Out_of_Distribution_Datasets-green) ![](https://img.shields.io/badge/Out_of_Distribution-orange)

  *Yun-Yun Tsai, Chengzhi Mao, Yow-Kuan Lin, Junfeng Yang.* [[Paper](https://arxiv.org/abs/2303.00198)][Code]
 
 - **Multimodal Prompting with Missing Modalities for Visual Recognition,** CVPR 2023 (arXiv:2303.03369). ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Missing_Modalities-orange)
   
   *Yi-Lun Lee, Yi-Hsuan Tsai, Wei-Chen Chiu, Chen-Yu Lee.* [[Paper](https://arxiv.org/abs/2303.03369)][[Code](https://github.com/YiLunLee/Missing_aware_prompts)]
 
 - **From Visual Prompt Learning to Zero-Shot Transfer: Mapping Is All You Need,** arXiv:2303.05266. ![](https://img.shields.io/badge/SEMAP-blue) ![](https://img.shields.io/badge/Image_Classification-green)
   
   *Ziqing Yang, Zeyang Sha, Michael Backes, Yang Zhang.* [[Paper](https://arxiv.org/abs/2303.05266)][Code]

 - **Diversity-Aware Meta Visual Prompting,** CVPR 2023 (arXiv:2303.08138). ![](https://img.shields.io/badge/DAMVP-blue) ![](https://img.shields.io/badge/Image_Classification-green)
   
   *Qidong Huang, Xiaoyi Dong, Dongdong Chen, Weiming Zhang, Feifei Wang, Gang Hua, Nenghai Yu.* [[Paper](https://arxiv.org/abs/2303.08138)][[Code](https://github.com/shikiw/DAM-VP)]
  
 - **Patch-Token Aligned Bayesian Prompt Learning for Vision-Language Models,** arXiv:2303.09100. ![](https://img.shields.io/badge/PBPrompt-blue) ![](https://img.shields.io/badge/Image_Classification-green)
   
   *Xinyang Liu, Dongsheng Wang, Miaoge Li, Zhibin Duan, Yishi Xu, Bo Chen, Mingyuan Zhou.* [[Paper](https://arxiv.org/abs/2303.09100)][Code]
   
 - **LION: Implicit Vision Prompt Tuning,** arXiv:2303.09992. ![](https://img.shields.io/badge/LION-blue) ![](https://img.shields.io/badge/Image_Classification-green)
   
   *Haixin Wang, Jianlong Chang, Xiao Luo, Jinan Sun, Zhouchen Lin, Qi Tian.* [[Paper](https://arxiv.org/abs/2303.09992)][Code]
   
 - **Fine-Grained Regional Prompt Tuning for Visual Abductive Reasoning,** arXiv:2303.10428. ![](https://img.shields.io/badge/RGP-blue) ![](https://img.shields.io/badge/Visual_Abductive_Reasoning-green)
   
   *Hao Zhang, Basura Fernando.* [[Paper](https://arxiv.org/abs/2303.10428)][Code]

 - **Visual Prompt Multi-Modal Tracking,** CVPR 2023 (arXiv:2303.10826). ![](https://img.shields.io/badge/ViPT-blue) ![](https://img.shields.io/badge/Tracking-green)
   
   *Jiawen Zhu, Simiao Lai, Xin Chen, Dong Wang, Huchuan Lu.* [[Paper](https://arxiv.org/abs/2303.10826)][Code]
   
 - **Explicit Visual Prompting for Low-Level Structure Segmentations,** CVPR 2023 (arXiv:2303.10883). ![](https://img.shields.io/badge/EVP-blue) ![](https://img.shields.io/badge/Low_Level_Structure_Segmentation-green)
   
   *Weihuang Liu, Xi Shen, Chi-Man Pun, Xiaodong Cun.* [[Paper](https://arxiv.org/abs/2303.10826)][[Code](https://github.com/NiFangBaAGe/Explict-Visual-Prompt)]

 - **CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition,** arXiv:2303.11313. ![](https://img.shields.io/badge/CG3D-blue) ![](https://img.shields.io/badge/3D_Recognition-green)
   
   *Deepti Hegde, Jeya Maria Jose Valanarasu, Vishal M. Patel.* [[Paper](https://arxiv.org/abs/2303.11313)][[Code](https://github.com/deeptibhegde/CLIP-goes-3D)]
   
   Comments: This works' idea is similar to our [Text4Point](https://arxiv.org/abs/2301.07584). 

 - **Multi-modal Prompting for Low-Shot Temporal Action Localization,** arXiv:2303.11732. ![](https://img.shields.io/badge/Temporal_Action_Localization-green)
   
   *Chen Ju, Zeqian Li, Peisen Zhao, Ya Zhang, Xiaopeng Zhang, Qi Tian, Yanfeng Wang, Weidi Xie.* [[Paper](https://arxiv.org/abs/2303.11732)][Code]
   
   Highlight: Enrich the meaning of an action class by querying the large-scale language model to give a detailed action description.
 
 - **Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning,** arXiv:2303.15230. ![](https://img.shields.io/badge/Troika-blue) ![](https://img.shields.io/badge/Compositional_Learning-green)
   
   *Siteng Huang, Biao Gong, Yutong Feng, Yiliang Lv, Donglin Wang.* [[Paper](https://arxiv.org/abs/2303.15230)][Code]
   
 - **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention,** arXiv:2303.16199. ![](https://img.shields.io/badge/LLaMA_Adapter-blue) ![](https://img.shields.io/badge/ChatBot-green)
   
   *Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, Yu Qiao.* [[Paper](https://arxiv.org/abs/2303.16199)][[Code](https://github.com/ZrrSkywalker/LLaMA-Adapter)]
   
   Highlight: Tuning the LLaMA(7B Params) to an excellent ChatBot with only 1.2M trainable parameters and 1 hour fine-tuning.

 - **Probabilistic Prompt Learning for Dense Prediction,** arXiv:2304.00779. ![](https://img.shields.io/badge/Dense_Prediction-green)
   
   *Hyeongjun Kwon, Taeyong Song, Somi Jeong, Jin Kim, Jinhyun Jang, Kwanghoon Sohn.* [[Paper](https://arxiv.org/abs/2304.00779)]
   
 - **Zero-shot Generative Model Adaptation via Image-specific Prompt Learning,** arXiv:2304.03119. ![](https://img.shields.io/badge/IPL-blue) ![](https://img.shields.io/badge/Image_Synthesis-green)
   
   *Jiayi Guo, Chaofei Wang, You Wu, Eric Zhang, Kai Wang, Xingqian Xu, Shiji Song, Humphrey Shi, Gao Huang.* [[Paper](https://arxiv.org/abs/2304.03119)][[Code](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation)]

 - **Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models,** arXiv:2304.07221. ![](https://img.shields.io/badge/IDPT-blue) ![](https://img.shields.io/badge/Point_Cloud-green)
   
   *Yaohua Zha, Jinpeng Wang, Tao Dai, Bin Chen, Zhi Wang, Shu-Tao Xia.* [[Paper](https://arxiv.org/abs/2304.07221)][[Code](https://github.com/zyh16143998882/IDPT)]

 - **PVP: Pre-trained Visual Parameter-Efficient Tuning,** arXiv:2304.13639. ![](https://img.shields.io/badge/PVP-blue) ![](https://img.shields.io/badge/Image_Classification-green)
   
   *Zhao Song, Ke Yang, Naiyang Guan, Junjie Zhu, Peng Qiao, Qingyong Hu.* [[Paper](https://arxiv.org/abs/2304.13639)][Code]

 - **Approximated Prompt Tuning for Vision-Language Pre-trained Models,** arXiv:2306.15706. ![](https://img.shields.io/badge/APT-blue) ![](https://img.shields.io/badge/VQA_,_Visual_Reasoning_,_Retrieval-green)
   
   *Qiong Wu, Shubin Huang, Yiyi Zhou, Pingyang Dai, Annan Shu, Guannan Jiang, Rongrong Ji.* [[Paper](https://arxiv.org/abs/2306.15706)][Code]

 - **Parameter-efficient Tuning of Large-scale Multimodal Foundation Model,** arXiv:2305.08381. ![](https://img.shields.io/badge/Aurora-blue) ![](https://img.shields.io/badge/VQA_,_Image_Text_Retrieval_,_Video_Text_Retrieval-green)
   
   *Haixin Wang, Xinlong Yang, Jianlong Chang, Dian Jin, Jinan Sun, Shikun Zhang, Xiao Luo, Qi Tian.* [[Paper](https://arxiv.org/abs/2305.08381)][[Code](https://github.com/WillDreamer/Aurora)]

  - **Hierarchical Prompt Learning for Multi-Task Learning,**  CVPR 2023 ![](https://img.shields.io/badge/Image_Classification-green)

    *Yajing Liu, Yuning Lu, Hao Liu, Yaozu An, Zhuoran Xu, Zhuokun Yao.* [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Hierarchical_Prompt_Learning_for_Multi-Task_Learning_CVPR_2023_paper.pdf)][Code]

  - **Focus Your Attention when Few-Shot Classification,** NeurIPS 2023 ![](https://img.shields.io/badge/Image_Classification-green)
    
    *Haoqing Wang, Shibo Jie, Zhi-Hong Deng.* [[Paper](https://nips.cc/media/neurips-2023/Slides/70162.pdf)][[Code](https://github.com/Haoqing-Wang/FORT)]



### Adapter

- **VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks,** CVPR 2022 (arXiv:2112.06825). 

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2112.06825)][[Code](https://github.com/ylsung/VL_adapter)] ![](https://img.shields.io/badge/VL_Adapter-blue) ![](https://img.shields.io/badge/Image_and_Video_QA,_Caption,_Visual_Reasoning-green) ![](https://img.shields.io/badge/MultiTask_Learning-orange)

- **AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition,** NeurIPS 2022 (arXiv:2205.13535). ![](https://img.shields.io/badge/AdaptFormer-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Temporal_Modeling-orange)

  *Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo.* [[Paper](https://arxiv.org/abs/2205.13535)][[Code](https://github.com/ShoufaChen/AdaptFormer)]

- **Zero-Shot Video Question Answering via Frozen Bidirectional Language Models,** NeurIPS 2022 (arXiv:2206.08155). ![](https://img.shields.io/badge/FrozenBiLM-blue) ![](https://img.shields.io/badge/VQA-green)

  *Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, Cordelia Schmid.* [[Paper](https://arxiv.org/abs/2206.08155)][[Code](https://github.com/antoyang/FrozenBiLM)]

- **ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning,** NeurIPS 2022 (arXiv:2206.13559). ![](https://img.shields.io/badge/ST_Adapter-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Temporal_Modeling-orange)

  *Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, Hongsheng Li.* [[Paper](https://arxiv.org/abs/2206.13559)][[Code](https://github.com/linziyi96/st-adapter)]

- **Convolutional Bypasses Are Better Vision Transformer Adapters,** arXiv:2207.07039. ![](https://img.shields.io/badge/Convpass-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Convolution_Inductive_Bias-orange)

  *Shibo Jie, Zhi-Hong Deng.* [[Paper](https://arxiv.org/abs/2207.07039)][[Code](https://github.com/JieShibo/PETL-ViT)]

- **Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets,** arXiv:2208.07463. ![](https://img.shields.io/badge/Conv_Adapter-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Design_for_ConvNet-orange)

  *Hao Chen, Ran Tao, Han Zhang, Yidong Wang, Wei Ye, Jindong Wang, Guosheng Hu, Marios Savvides.* [[Paper](https://arxiv.org/abs/2208.07463)][Code]

- **Effective Adaptation in Multi-Task Co-Training for Unified Autonomous Driving,** NeurIPS 2022 (arXiv:2209.08953). ![](https://img.shields.io/badge/LV_Adapter-blue) ![](https://img.shields.io/badge/Detection,_Segmentation-green)

  *Xiwen Liang, Yangxin Wu, Jianhua Han, Hang Xu, Chunjing Xu, Xiaodan Liang.* [[Paper](https://arxiv.org/abs/2209.08953)][Code]

- **Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks,** NeurIPS 2022 (arXiv:2210.03265). ![](https://img.shields.io/badge/Polyhistor-blue) ![](https://img.shields.io/badge/Dense_Vision_Tasks-green) ![](https://img.shields.io/badge/MultiTask_Learning-orange)

  *Yen-Cheng Liu, Chih-Yao Ma, Junjiao Tian, Zijian He, Zsolt Kira.* [[Paper](https://arxiv.org/abs/2210.03265)][Code]

- **SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models,** arXiv:2210.03794. ![](https://img.shields.io/badge/SVL_Adapter-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Omiros Pantazis, Gabriel Brostow, Kate Jones, Oisin Mac Aodha.* [[Paper](https://arxiv.org/abs/2210.03794)][[Code](https://github.com/omipan/svl_adapter)]

- **Cross-Modal Adapter for Text-Video Retrieval,** arXiv:2211.09623. ![](https://img.shields.io/badge/CM_Adapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Haojun Jiang, Jianke Zhang, Rui Huang, Chunjiang Ge, Zanlin Ni, Jiwen Lu, Jie Zhou, Shiji Song, Gao Huang.* [[Paper](https://arxiv.org/abs/2211.09623)][[Code](https://github.com/LeapLabTHU/Cross-Modal-Adapter)]

- **Vision Transformers are Parameter-Efficient Audio-Visual Learners,** arXiv:2212.07983. ![](https://img.shields.io/badge/LAVISH-blue) ![](https://img.shields.io/badge/Audio_Visual_Tasks-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Yan-Bo Lin, Yi-Lin Sung, Jie Lei, Mohit Bansal, Gedas Bertasius.* [[Paper](https://arxiv.org/abs/2212.07983)][[Code](https://github.com/GenjiB/LAVISH)]

  Take away message: Pre-trained vision transformer can deal with audio data by representing 1D raw audio signal as 2D audio image.

- **Multimodal Video Adapter for Parameter Efficient Video Text Retrieval,** arXiv:2301.07868. ![](https://img.shields.io/badge/MV-Adapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Bowen Zhang, Xiaojie Jin, Weibo Gong, Kai Xu, Zhao Zhang, Peng Wang, Xiaohui Shen, Jiashi Feng.* [[Paper](https://arxiv.org/abs/2301.07868)][Code]

- **AIM: Adapting Image Models for Efficient Video Action Recognition,** ICLR 2023 (arXiv:2302.03024). ![](https://img.shields.io/badge/AIM-blue) ![](https://img.shields.io/badge/Action_Recognition-green) ![](https://img.shields.io/badge/Image2Video-orange)

  *Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, Mu Li.* [[Paper](https://arxiv.org/abs/2302.03024)][[Code](https://github.com/taoyang1122/adapt-image-models)]

- **Offsite-Tuning: Transfer Learning without Full Model,** arXiv:2302.04870. ![](https://img.shields.io/badge/Offsite_Tuning-blue) ![](https://img.shields.io/badge/Both_Vision_and_Language_Tasks-green) ![](https://img.shields.io/badge/Privacy-orange)

  *Guangxuan Xiao, Ji Lin, Song Han.* [[Paper](https://arxiv.org/abs/2302.04870)][[Code](https://github.com/mit-han-lab/offsite-tuning)]

- **UniAdapter: Unified Parameter-Efficient Transfer Learning for Cross-modal Modeling,** arXiv:2302.06605. ![](https://img.shields.io/badge/UniAdapter-blue) ![](https://img.shields.io/badge/Text_Video_Retrieval,_Text_Image_Retrieval,_VideoQA,_VQA-green) ![](https://img.shields.io/badge/CrossModal-orange)

  *Haoyu Lu, Mingyu Ding, Yuqi Huo, Guoxing Yang, Zhiwu Lu, Masayoshi Tomizuka, Wei Zhan.* [[Paper](https://arxiv.org/abs/2302.06605)][[Code](https://github.com/RERV/UniAdapter)]

- **Towards Efficient Visual Adaption via Structural Re-parameterization,** arXiv:2302.08106. ![](https://img.shields.io/badge/RepAdapter-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Re-parameterization-orange)

  *Gen Luo, Minglang Huang, Yiyi Zhou, Xiaoshuai Sun, Guannan Jiang, Zhiyu Wang, Rongrong Ji.* [[Paper](https://arxiv.org/abs/2302.08106)][[Code](https://github.com/luogen1996/RepAdapter)]

- **T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models,** arXiv:2302.08453. ![](https://img.shields.io/badge/T2I_Adapter-blue) ![](https://img.shields.io/badge/Image_Generation-green) ![](https://img.shields.io/badge/Diffusion_Model-orange)

  *Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, Xiaohu Qie.* [[Paper](https://arxiv.org/abs/2302.08453)][[Code](https://github.com/TencentARC/T2I-Adapter)]

- **kNN-Adapter: Efficient Domain Adaptation for Black-Box Language Models,** arXiv:2302.10879. ![](https://img.shields.io/badge/kNN_Adapter-blue) ![](https://img.shields.io/badge/Domain_Adaptation-green) ![](https://img.shields.io/badge/Black_Box_Language_Model-orange)

  *Yangsibo Huang, Daogao Liu, Zexuan Zhong, Weijia Shi, Yin Tat Lee.* [[Paper](https://arxiv.org/abs/2302.10879)][Code]

- **Side Adapter Network for Open-Vocabulary Semantic Segmentation,** arXiv:2302.12242. ![](https://img.shields.io/badge/Side_Adapter_Network-blue) ![](https://img.shields.io/badge/Segmentation-green) ![](https://img.shields.io/badge/Open_Vocabulary-orange)

  *Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, Xiang Bai.* [[Paper](https://arxiv.org/abs/2302.12242)][Code]

 - **Dual-path Adaptation from Image to Video Transformers,** arXiv:2303.09857. ![](https://img.shields.io/badge/DualPath-blue) ![](https://img.shields.io/badge/Action_Recognition-green)
   
   *Jungin Park, Jiyoung Lee, Kwanghoon Sohn.* [[Paper](https://arxiv.org/abs/2303.09857)][[Code](https://github.com/park-jungin/DualPath)]
   
   Highlight: Modeling temporal information in a separate path.

 - **Contrastive Alignment of Vision to Language Through Parameter-Efficient Transfer Learning,** ICLR 2023 (arXiv:2303.11866). ![](https://img.shields.io/badge/LilT-blue) ![](https://img.shields.io/badge/Retrieval_and_Classification-green)
   
   *Zaid Khan, Yun Fu.* [[Paper](https://arxiv.org/abs/2303.11866)][[Code](https://github.com/codezakh/LilT)]
   
   Highlight: Aligning an already-trained vision and language model with adapter.
   
 - **A Closer Look at Parameter-Efficient Tuning in Diffusion Models,** arXiv:2303.18181. ![](https://img.shields.io/badge/Diffusion_Model-green)
   
   *Chendong Xiang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu.* [[Paper](https://arxiv.org/abs/2303.18181)]
   
 - **SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**, EMNLP 2022 (arXiv:2210.04284). ![](https://img.shields.io/badge/-SparseAdapter-blue) ![](https://img.shields.io/badge/-GLUE%20Benchmark-green) ![](https://img.shields.io/badge/-Pretrained%20Language%20Model-orange)
 
   *Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao.* [[Paper](https://arxiv.org/pdf/2210.04284.pdf)][[Code](https://github.com/Shwai-He/SparseAdapter)]
  
 - **Parameter-efficient Model Adaptation for Vision Transformers**, AAAI 2023 (arXiv:2203.16329). ![](https://img.shields.io/badge/PEViT-blue) ![](https://img.shields.io/badge/-Image_Classification-green) ![](https://img.shields.io/badge/-KAdaptation-orange)
  
   *Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, Xin Eric Wang.* [[Paper](https://arxiv.org/abs/2203.16329)][[Code](https://github.com/eric-ai-lab/PEViT)]

 - **TaCA: Upgrading Your Visual Foundation Model with Task-agnostic Compatible Adapter**, arXiv:2306.12642. ![](https://img.shields.io/badge/-TaCA-blue) ![](https://img.shields.io/badge/-Video_Text_Retrieval_,_Video_Recognition_,_Visual_Question_Answering-green) 
  
   *Binjie Zhang, Yixiao Ge, Xuyuan Xu, Ying Shan, Mike Zheng Shou.* [[Paper](https://arxiv.org/abs/2306.12642)][[Code](https://github.com/TencentARC/TaCA)]

- **Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis**, CVPR 2024.  ![](https://img.shields.io/badge/DAPT-blue) ![](https://img.shields.io/badge/Point_Cloud-green) ![](https://img.shields.io/badge/Adapter_with_Prompt-orange)

  *Xin Zhou , Dingkang Liang , Wei Xu, Xingkui Zhu ,Yihan Xu, Zhikang Zou, Xiang Bai.* [[Paper](https://arxiv.org/abs/2403.01439)][[Code](https://github.com/LMD0311/DAPT)]

### Unified

- **Towards a Unified View of Parameter-Efficient Transfer Learning,** ICLR 2022 (arXiv:2110.04366). ![](https://img.shields.io/badge/Translation,_Summarization,_Language_Understanding,_Text_Classification-green) ![](https://img.shields.io/badge/Unified_View-orange)

  *Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig.* [[Paper](https://arxiv.org/abs/2110.04366)][[Code](https://github.com/jxhe/unify-parameter-efficient-tuning)]

- **Neural Prompt Search,** arXiv:2206.04673. ![](https://img.shields.io/badge/NOAH-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Unified_Search_Framework-orange)

  *Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu.* [[Paper](https://arxiv.org/abs/2206.04673)][[Code](https://github.com/ZhangYuanhan-AI/NOAH)]

- **AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning,** arXiv:2301.12132. ![](https://img.shields.io/badge/AutoPEFT-blue) ![](https://img.shields.io/badge/_Text_Classification-green) ![](https://img.shields.io/badge/Unified_Search_Framework-orange)

  *Han Zhou, Xingchen Wan, Ivan Vulić, Anna Korhonen.* [[Paper](https://arxiv.org/abs/2301.12132)][[Code](https://github.com/cambridgeltl/autopeft)]

- **Rethinking Efficient Tuning Methods from a Unified Perspective,** arXiv:2303.00690. ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Unified_View-orange)

  *Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Yiliang Lv, Deli Zhao, Jingren Zhou.* [[Paper](https://arxiv.org/abs/2303.00690)][Code]

- **VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control,** arXiv:2308.09804. ![](https://img.shields.io/badge/VL_PET-blue)![](https://img.shields.io/badge/VQA_,_Image_Caption_,_Visual_Reasoning_,_Video_Caption-green) ![](https://img.shields.io/badge/Granularity_Control-orange)

  *Zi-Yuan Hu, Yanyang Li, Michael R. Lyu, Liwei Wang.* [[Paper](https://arxiv.org/abs/2308.09804)][[Code](https://github.com/HenryHZY/VL-PET)]

### Others

- Check out [thunlp/DeltaPapers](https://github.com/thunlp/DeltaPapers) if you are interested in the progress of NLP domain.

- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning,** NeurIPS 2022 (arXiv:2206.06522). ![](https://img.shields.io/badge/LST-blue) ![](https://img.shields.io/badge/GLUE,_VQA,_Visual_Reasoning,_Image_Caption-green) ![](https://img.shields.io/badge/Side_Tuning-orange)

  *Yi-Lin Sung, Jaemin Cho, Mohit Bansal.* [[Paper](https://arxiv.org/abs/2206.06522)][[Code](https://github.com/ylsung/Ladder-Side-Tuning)]

- **Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning,** NeurIPS 2022 (arXiv:2210.08823). ![](https://img.shields.io/badge/SSF-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Feature_Scale_&_Shift-orange)

  *Dongze Lian, Daquan Zhou, Jiashi Feng, Xinchao Wang.* [[Paper](https://arxiv.org/abs/2210.08823)][[Code](https://github.com/dongzelian/SSF)]

- **FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer,** AAAI 2023 (arXiv:2212.03145). ![](https://img.shields.io/badge/FacT-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Tensor_Decomposition-orange)

  *Shibo Jie, Zhi-Hong Deng.* [[Paper](https://arxiv.org/abs/2212.03145)][[Code](https://github.com/JieShibo/PETL-ViT)]

- **Important Channel Tuning,** Openreview. ![](https://img.shields.io/badge/ICT-blue) ![](https://img.shields.io/badge/Image_Classification-green)

  *Hengyuan Zhao, Pichao WANG, Yuyang Zhao, Fan Wang, Mike Zheng Shou.* [[Paper](https://openreview.net/forum?id=TTMyoOdB9hZ)][Code]

- **MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering,** arXiv:2303.01239. ![](https://img.shields.io/badge/MixPHM-blue) ![](https://img.shields.io/badge/VQA-green)

  *Jingjing Jiang, Nanning Zheng.* [[Paper](https://arxiv.org/abs/2303.01239)][Code]

- **Revisit Parameter-Efficient Transfer Learning: A Two-Stage Paradigm,** arXiv:2303.07910. ![](https://img.shields.io/badge/Image_Classification-green)

  *Hengyuan Zhao, Hao Luo, Yuyang Zhao, Pichao Wang, Fan Wang, Mike Zheng Shou.* [[Paper](https://arxiv.org/abs/2303.07910)][Code]

- **Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning,** CVPR 2023 (arXiv2212.03220) ![](https://img.shields.io/badge/Image_Classification-green)

  *Cheng-Hao Tu, Zheda Mai, Wei-Lun Chao.*  [[Paper](https://arxiv.org/abs/2212.03220)][[Code](https://github.com/andytu28/VQT)]

- **DTL: Disentangled Transfer Learning for Visual Recognition,** AAAI 2024 (arXiv:2312.07856). ![](https://img.shields.io/badge/DTL-blue) ![](https://img.shields.io/badge/Image_Classification-green) ![](https://img.shields.io/badge/Disentangled-orange)

  *Minghao Fu, Ke Zhu, Jianxin Wu.* [[Paper](https://arxiv.org/abs/2312.07856)][Code]
  
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



