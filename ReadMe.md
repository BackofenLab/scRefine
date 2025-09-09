scRefine

**Motivation** Variational autoencoders (VAEs) have become useful tools for learning meaningful latent representations of high-dimensional biological data, making them particularly useful for analyzing single-cell transcriptomic profiles. In the context of immune cell data, scVAEs can capture complex cellular heterogeneity, enabling the discovery of rare immune subpopulations and the identification of subtle transcriptional shifts during immune responses. 
Given the highly dynamic nature of immune cells, which transition between states in response to pathogens, inflammation, or disease, traditional clustering methods often struggle to capture continuous trajectories and subtle variations. By leveraging scVAEs, we can learn low-dimensional representations that preserve biologically relevant structures, facilitating better characterization of immune cell states and their functional roles. Additionally we introduce a novel approach to integrating data from different batches. 


**running scRefine**

**data preparation**
see preprocess.py

**training hvae**
see train_hvae.py

**contrastive learning**
see train_simclr.py
