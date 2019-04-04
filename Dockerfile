FROM jupyter/minimal-notebook

#RUN sudo apt-get install python3-pip
#RUN echo $(which python3)
RUN git clone https://github.com/mtmoncur/deepracer_env.git && \
    cd deepracer_env && \
    pip install -e . && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
