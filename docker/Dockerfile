FROM python:3.6

MAINTAINER lferiani <l.feriani@lms.mrc.ac.uk>

# Set an environment variable to fix:
# libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1
# more env variables
ENV CODE_PATH="/tierpsy-tracker"
ENV DOCKER_HOME="/home/tierpsy_user"
ENV PATH="${PATH}:/home/tierpsy_user/.local/bin"
ENV XDG_RUNTIME_DIR="/tmp/runtime-tierpsy_user"

# Install Python 3, PyQt5
ARG DEBIAN_FRONTEND noninteractive
# RUN apt-get update && apt-get install -y python3.6 python3-pip git ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg
RUN apt-get install -y libsm6 libxrender1 libfontconfig1
RUN python3 -m pip install --upgrade pip setuptools

# add user, make a few folders, give ownership to user
RUN adduser --quiet --disabled-password tierpsy_user
RUN mkdir ${CODE_PATH} && chown -R tierpsy_user ${CODE_PATH}
# create mountpoints for the local and network drives.
RUN mkdir /DATA /DATA/network_drive /DATA/local_drive && \
    chown -R tierpsy_user /DATA
# prevent a warning about XDG_RUNTIME_DIR not being set.
RUN mkdir /tmp/runtime-tierpsy_user && \
    chown tierpsy_user /tmp/runtime-tierpsy_user && \
    chmod 0700 /tmp/runtime-tierpsy_user

# get the source code
# git clone is unreliable as it can be cached, COPY overrides caching
# .dockerfile is necessary to only copy what's not being ignored by git
COPY --chown=tierpsy_user:tierpsy_user . ${CODE_PATH}
# fix permissions
# RUN chown -R tierpsy_user:tierpsy_user ${CODE_PATH} && \
RUN chmod -R 755 ${CODE_PATH}

USER tierpsy_user
RUN ls -lah ${CODE_PATH}/*

# install dependencies and code
RUN pip3 install -r ${CODE_PATH}/docker/docker_requirements.txt
RUN pip3 install -e ${CODE_PATH}

# get food contour model
RUN wget https://zenodo.org/record/5575338/files/unet_RMSprop-5-04999-0.3997.h5 \
    -P ${CODE_PATH}/tierpsy/extras/

# edit .bashrc:
# python => python3, script to check if tierpsy is running, splash screen
RUN echo "alias python=python3" >> ${DOCKER_HOME}/.bashrc
RUN echo "alias is_tierpsy_running=${CODE_PATH}/docker/is_tierpsy_running.sh" \
    >> ~/.bashrc
RUN echo "clear" >> ${DOCKER_HOME}/.bashrc
RUN echo "cat ${CODE_PATH}/docker/tierpsy_splash.txt" >> ${DOCKER_HOME}/.bashrc
RUN echo "tierpsy_gui &" >> ${DOCKER_HOME}/.bashrc

# make it so that tierpsy opens in the /DATA folder
# hopefully this works with github actions
WORKDIR /DATA

# launch app
CMD ["bash"]