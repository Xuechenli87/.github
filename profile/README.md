# 深圳大学计算机视觉研究所 :sunny: 
## 研究领域
   1. 通用深度学习理论
    
      - 新型网络架构、特征提取模块提升CNN 性能；提出小波卷积层代替下采样池化及上采样模块，解决高低频信号混叠问题。
        - **WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for Noise-Robust Image Classification (IEEE TIP 2021)** [[paper](https://ieeexplore.ieee.org/document/9508165)]         
        - **Wavelet Integrated CNNs for Noise-Robust Image Classification (CVPR 2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)] [[code](https://github.com/CVI-SZU/WaveCNet)]
     	 - **Neuron segmentation using 3D wavelet integrated encoder–decoder network (Bioinformatics) 38(3), 2022, 809–817** [[paper](https://academic.oup.com/bioinformatics/article/38/3/809/6396862?login=true)][[code](https://github.com/LiQiufu/3DWaveUNet)]
      - 结合手工特征，引导网络早期的快速收敛。
        - **Adaptive Weighting of Hand-crafted Feature Losses for Facial Expression Recognition (IEEE TCYB 2021)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/TCYB2019.pdf)] [[project](https://wcxie.github.io/Weicheng-Xie/PaperProject/AWHFL_TCYB2019/)]
      - 设计了忆阻模糊深度学习系统，以在低精度的设备条件下提高图像识别精度。以及基于忆阻的量化深度学习网络结构，从而加快训练速度和减小存储空间。
        - **Memristive Fuzzy Deep Learning Systems (IEEE TFS 2021)** [[paper](https://ieeexplore.ieee.org/document/9098057)] 
        - **Memristive Quantized Neural Networks A Novel Approach to Accelerate Deep Learning On-Chip (IEEE TCYB 2021)** [[paper](https://ieeexplore.ieee.org/document/8705375)] 
      - 特征的自适应分组抑制、激活与基于特征密度的失活算法
        - **Group-wise Inhibition based Feature Regularization for Robust Classification (ICCV 2021)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/ICCV2021.pdf)] [[project](https://csse.szu.edu.cn/pages/research/details?id=10)] [[code](https://github.com/LinusWu/TENET_Training)] 
        - **Group-Wise Dynamic Dropout Based on Latent Semantic Variation (AAAI 2020)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/AAAI2020.pdf)] [[project](https://csse.szu.edu.cn/pages/research/details?id=35)]
      - 通过特征的对抗攻击提升针对开放场景的有效性，并将噪声隐藏在高频区域提升不可察觉性
        - **Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity (CVPR 2022)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/CVPR2022-Luo.pdf)] [[code](https://github.com/LinQinLiang/SSAH-adversarial-attack)] 


  2. 弱监督物体检测/语义分割

     - 基于图像类别标签的物体检测网络、语义分割网络，减少通用物体检测/分割网络对标注数据的要求。研究如何利用大规模预训练模型减少网络对训练数据的要求，在零样本开放数据上依然能够取得良好性能。
       - **Geometry Constrained Weakly Supervised Object Localization (ECCV 2020)** [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710477.pdf)][[code](https://github.com/lwzeng/GC-Net)]
       - **CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation (CVPR 2022)** [[paper](https://arxiv.org/abs/2203.02668)] [[project](https://sierkinhane.github.io/clims/)] [[code](https://github.com/CVI-SZU/CLIMS)]
       - **CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation (CVPR 2022)** [[paper](https://arxiv.org/pdf/2203.13505.pdf)] [[project]()] [[code](https://github.com/CVI-SZU/CCAM)]
     - 基于Transformer的文本检测
       - **Self-attention based Text Knowledge Mining for Text Detection (CVPR 2021)** [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wan_Self-Attention_Based_Text_Knowledge_Mining_for_Text_Detection_CVPR_2021_paper.pdf)][[code](https://github.com/CVI-SZU/STKM)]
     - 基于场景一致性表示学习的视频场景分割
       - **Scene Consistency Representation Learning for Video Scene Segmentation (CVPR 2022)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/CVPR2022-Wu.pdf)] [[project](https://csse.szu.edu.cn/pages/research/details?id=195)] 

  3. 人脸识别/分析

     基于深度卷积网络的人脸识别、表情、年龄等属性识别方法及系统，基于人脸视频的个体性格、抑郁等心理情绪预测。开发的人脸识别系统在深圳高交会、双创周、杭州G20和新疆安防广泛使用。
     -  基于人脸时域变化特征提取与学习的抑郁、性格分析，基于个体人脸反应生成网络的性格分析
        - **Self-supervised Learning of Person-specific Facial Dynamics for Automatic Personality Recognition (IEEE TAFC 2021)** [[paper](https://ieeexplore.ieee.org/document/9373959)]
        - **Spectral Representation of Behaviour Primitives for Depression Analysis (IEEE TAFC 2020)** [[paper](https://ieeexplore.ieee.org/document/8976305)] [[code](https://github.com/SSYSteve/Human-behaviour-based-depression-analysis-using-hand-crafted-statistics-and-deep-learned)]
        - **Personality Recognition by Modelling Person-specific Cognitive Processes using Graph Representation (ACM MM 2021)** [[paper](https://dl.acm.org/doi/10.1145/3474085.3475460)] [[code](https://github.com/SSYSteve/Learning-Graph-Representation-of-Person-specific-Cognitive-Processes-from-Audio-visual-Behaviours-fo)]
     - 三维人脸点云识别与表情
       - **Orthogonalization Guided Feature Fusion Network for Multimodal 2D+3D Facial Expression Recognition (IEEE TMM 2021)** [[paper](https://ieeexplore.ieee.org/document/9115253)]
     - 二维人脸表情识别与分析、合成
       - **Triplet loss with multistage outlier suppression and class-pair margins for facial expression recognition (IEEE TCSVT 2022)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/TCSVT2022.pdf)] [[project](https://wcxie.github.io/Weicheng-Xie/PaperProject/TLWMOSCPM_TCSVT2021/)] 
       - **A Novel Transient Wrinkle Detection Algorithm and Its Application for Expression Synthesis (IEEE TMM 2017)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/TMM2017.pdf)] [[project](https://wcxie.github.io/Weicheng-Xie/PaperProject/TWDA_TMM2016/)]
       - **Surrogate Network-based Sparseness Hyper-parameter Optimization for Deep Expression Recognition (Pattern Recognition 2021)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/PR2021.pdf)] [[project](https://wcxie.github.io/Weicheng-Xie/PaperProject/SNSHO_PR2021/)] 
       - **Sparse deep feature learning for facial expression recognition (Pattern Recognition 2019)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/PR2019.pdf)] [[project](https://wcxie.github.io/Weicheng-Xie/PaperProject/SDFL_PR2019/)]
     - 基于多维边缘特征的面部运动单元的关系图学习以及动作单元的识别应用
       - **Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition (IJCAI 2022)** [[paper](https://wcxie.github.io/Weicheng-Xie/pdf/IJCAI2022.pdf)] [[code](https://github.com/CVI-SZU/MEFARG)] 

  4. 人脸生成与属性编辑

     - 基于图像翻译的人脸属性编辑
       - **AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation (IEEE TMM 2022)** [[paper](https://ieeexplore.ieee.org/document/9541089)]
       - **Translate the Facial Regions You Like Using Self-Adaptive Region Translation (AAAI 2021)** [[paper](https://www.aaai.org/AAAI21Papers/AAAI-1663.LiuW.pdf)]
       - **GazeFlow: Gaze Redirection with Normalizing Flows (IJCNN 2021)** [[paper](https://ieeexplore.ieee.org/abstract/document/9533913)][[code](https://github.com/CVI-SZU/GazeFlow)] [[demo](https://github.com/ackness/GazeFlow_demo)]
       - **Gated SwitchGAN for multi-domain facial image translation (IEEE TMM 2021)** [[paper](https://arxiv.org/pdf/2111.14096)]
       - **Deep Feature Consistent Variational Autoencoder (WACV 2017)** [[paper](https://arxiv.org/pdf/1610.00291.pdf)] [[project](https://houxianxu.github.io/assets/project/dfcvae)] [[code](https://github.com/houxianxu/DFC-VAE)]
     - 基于预训练模型的人脸属性编辑
       - **Lifelong Age Transformation with a Deep Generative Prior (IEEE TMM 2022)** [[paper](https://ieeexplore.ieee.org/abstract/document/9726897/)] [[project](https://houxianxu.github.io/assets/project/age-editing)]
       - **Guidedstyle: Attribute Knowledge Guided Style Manipulation for Semantic Face Editing (Neural Networks 2022)** [[paper](https://www.sciencedirect.com/science/article/pii/S0893608021004081)] [[project](https://houxianxu.github.io/assets/project/guided-style)]
       - **SSFlow: Style-guided Neural Spline Flows for Face Image Manipulation (ACM MM 2021)** [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475454)]
     - 基于文本的人脸生成与编辑
       - **TextFace: Text-to-Style Mapping based Face Generation and Manipulation (IEEE TMM 2022)** [[paper](https://ieeexplore.ieee.org/abstract/document/9737433/)] [[project](https://houxianxu.github.io/assets/project/textface)]

  5. 医学图像智能分析

     - 细胞检测、分割与分类
       - **Sample hardness based gradient loss for long-tailed cervical cell detection (MICCAI 2022)** 
       - **Deep Cross Residual Network for HEp2 Cell Staining Pattern Classification (Pattern Recognition 2018)** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320318301705)]
       - **HEp-2 Specimen Image Segmentation and Classification Using Very Deep Fully Convolutional Network (IEEE TMI 2017)** [[paper](https://ieeexplore.ieee.org/document/7862234/citations#citations)]
     - 肺部影像智能分析
       - **A Self-supervised Feature-standardization-block for Cross-domain Lung Disease Classification (Methods 2022)** [[paper](https://www.sciencedirect.com/science/article/pii/S104620232100133X)]
       - **Learning from Pseudo Lesion: A Self-supervised Framework for COVID-19 Diagnosis (Neural Computing and Applications 2022)** [[paper](https://arxiv.org/pdf/2106.12313.pdf)]
       - **Multi-resolution convolutional networks for chest X-ray radiograph based lung nodule detection (Artificial Intelligence in Medicine 2020)** [[paper](https://www.sciencedirect.com/science/article/pii/S093336571830633X)]
       - **A Solitary Feature-based Lung Nodule Detection Approach for Chest X-Ray Radiographs (IEEE JBHI 2018 封面亮点)** [[paper](https://ieeexplore.ieee.org/document/7837683)]
     - 大脑神经元分割重建
       - **Deep learning based neuronal soma detection and counting for Alzheimer's disease analysis (Computer Methods and Programs in Biomedicine 2021)** [[paper](https://www.sciencedirect.com/science/article/pii/S0169260721000985)]
       - **3D Neuron Reconstruction in Tangled Neuronal Image with Deep Networks (IEEE TMI 2020)** [[paper](https://ieeexplore.ieee.org/document/8758392)]
     - 病理图片细胞分割与分析
       - **Instance-aware Self-supervised Learning for Nuclei Segmentation (MICCAI 2020)** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-59722-1_33)]
       - **MI2GAN: Generative Adversarial Network for Medical Image Domain Adaptation using Mutual Information Constraint (MICCAI 2020)** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_50)]
       - **Reverse active learning based atrous DenseNet for pathological image classification (BMC Bioinformatics 2019)** [[paper](https://link.springer.com/article/10.1186/s12859-019-2979-y)]
     - 皮肤病变区域分割
       - **Dense Encoder-Decoder–Based Architecture for Skin Lesion Segmentation (Cognitive Computation 2021)** [[paper](https://link.springer.com/article/10.1007/s12559-020-09805-6)]
       - **Skin lesion analysis towards melanoma detection using deep learning network (Sensors 2018 ESI高被引论文)** [[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5855504/)]
     - B超图像甲状腺结节分割
       - **Boundary Regression based deep neural network for thyroid nodule segmentation in ultrasound images (Neural Computing and Applications 2022)**
     - 眼部OCT图像分析
       - **Deep Learning Based Early Stage Diabetic Retinopathy Detection Using Optical Coherence Tomography (Neurocomputing 2019)** [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219312172)]
