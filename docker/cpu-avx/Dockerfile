# Version: 2.0.0
FROM registry.baidubce.com/paddlepaddle/paddle:2.1.0

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

RUN pip3.7 install paddlehub==2.1.0 -i https://mirror.baidu.com/pypi/simple

RUN git clone -b release/2.1 https://gitee.com/paddlepaddle/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN mkdir -p /PaddleOCR/inference/

ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_det_infer.tar -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_cls_infer to ch_ppocr_mobile_v2.0_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /PaddleOCR/inference/

ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_rec_infer.tar -C /PaddleOCR/inference/

# fix bug
RUN sed -i '44c if sys.version_info[:2] < (3, 8):' /usr/local/python3.7.0/lib/python3.7/site-packages/git/config.py

EXPOSE 8868

CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]
