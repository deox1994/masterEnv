# Specify the base image
ARG BASE_IMAGE=image
FROM ${BASE_IMAGE}

# Install pyserial library for energy monitoring
RUN apt-get update
RUN pip install pyserial

# Include new modules in the task manager file
COPY ultralytics/Edge/nn/tasks.py /ultralytics/ultralytics/nn

# General V8DetectionLoss (Loss function + Weighted BCE) function:
COPY ultralytics/Edge/utils/loss.py /ultralytics/ultralytics/utils

# Additional Loss metrics (WIoU and MPDIoU)
COPY ultralytics/Edge/utils/metrics.py /ultralytics/ultralytics/utils

# New modules: block, attention, conv, pooling
COPY modules /ultralytics/modules

# Trained models
COPY models /ultralytics/models

# Change dataset directory in settings.json
COPY ultralytics/.config/settings.json /root/.config/Ultralytics

# Add train script
COPY train.py /ultralytics/

# Add export script
COPY export.py /ultralytics/

# Add evalModel script
COPY evaluate.py /ultralytics/





