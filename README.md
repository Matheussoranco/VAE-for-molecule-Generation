# VAE-for-molecule-Generation
The model described in the paper Automatic chemical design using a data-driven continuous representation of molecules generates new molecules via efficient exploration of open-ended spaces of chemical compounds. The model consists of three components: Encoder, Decoder and Predictor. The Encoder converts the discrete representation of a molecule into a real-valued continuous vector, and the Decoder converts these continuous vectors back to discrete molecule representations. The Predictor estimates chemical properties from the latent continuous vector representation of the molecule. Continuous representations allow the use of gradient-based optimization to efficiently guide the search for optimized functional compounds.

#Sources
- Keras article: (https://keras.io/examples/generative/molecule_generation/)
- Automatic chemical design using a data-driven continuous representation of molecules: (https://arxiv.org/abs/1610.02415)
- MolGAN: An implicit generative model for small molecular graphs: (https://arxiv.org/abs/1805.11973)
