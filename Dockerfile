FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries while
RUN pip install changepoynt --no-deps
RUN pip install numpy scipy matplotlib pandas dash plotly h5py fbpca numba

# download the necessary data we need for the prototype
RUN apt-get update
RUN apt-get install wget -y
RUN wget https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/328266/HS2.zip

# install and run utility tool to unzip files
RUN apt-get install unzip -y
RUN unzip HS2.zip

# copy the app file into the container
COPY . ./

ENTRYPOINT ["python", "CPSELECT_Dash.py"]