# Installation

To install pytorch on linux, run:

    conda create -n dnabert python=3.9 -y && conda activate dnabert

    pip3 install torch torchvision torchaudio transformers

    wget 'https://unipiit-my.sharepoint.com/:u:/g/personal/m_tolloso_studenti_unipi_it/EZLI20s-jLhLvx9SGRLmpEABYzh5QBkV4cSl-wfbVgfu2w?e=kT998J&download=1' -O pytorch_model.bin
    mv pytorch_model.bin dnabert/pytorch_model.bin
