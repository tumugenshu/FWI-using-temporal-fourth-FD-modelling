# FWI-using-temporal-fourth-FD-modelling
These code are used to complete elastic full waveform inversion using temporal fourth FD modelling. In order to accelete the algorith, GPU-based versions are privided here. What's more, the GPU shared memory is used in block level and the efficiency improvement is about 10%~15% on GTX 750ti, where efficiency improvement is dependent on the length of FD orders.