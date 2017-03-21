# noise_reduction

> See test results on: [https://dodiku.github.io/noise_reduction/](https://dodiku.github.io/noise_reduction/)

## Audio enhancements feature tests in Python3

#### Installation
To install:
1. ``$ brew install sox``
1. ``$ brew install vorbis-tools``
1. Create a virtualenv
1. Install dependencies in one of two options:
  - manually *(recommended)*:  
      ``$ pip3 install librosa``  
      ``$ pip3 install pysndfx``

  - or automatically using pip:  
      ``$ pip3 install -r requirements.txt``

To run:  
``$ python3 noise.py``


#### Interesting resources:
- LibROSA ([documentation](http://librosa.github.io/librosa/index.html) + [repository](https://github.com/librosa/librosa) + [paper](https://bmcfee.github.io/papers/scipy2015_librosa.pdf))
- Think DSP ([book](http://greenteapress.com/wp/think-dsp/) + [repository](https://github.com/AllenDowney/ThinkDSP/))
- Pyo ([blog post](http://www.matthieuamiguet.ch/blog/diy-guitar-effects-python) + [repository](https://github.com/belangeo/pyo))
- pysndfx ([repository](https://github.com/carlthome/python-audio-effects/tree/04dbee6063b0537b63346bb1e55deb03406e1170/pysndfx))

#### A bit less relevant papers:
- Noise Cancellation Method for Robust Speech Recognition ([PDF](http://research.ijcaonline.org/volume45/number11/pxc3879438.pdf))
- Robust Features for Noisy Speech Recognition using MFCC Computation from Magnitude Spectrum of Higher Order Autocorrelation Coefficients
([PDF](https://pdfs.semanticscholar.org/a483/5f28c02f07e6bef04ff9db948505dc990af7.pdf))
- Improving the Noise-Robustness of Mel-Frequency Cepstral Coefficients for Speech Processing
([PDF](http://www.sapaworkshops.org/2006/2006/papers/131.pdf))
