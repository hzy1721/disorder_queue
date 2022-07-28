# -*- coding : utf-8-*-
# coding=gbk
import datetime as dt
from kafka import KafkaProducer
import json


# producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'), api_version = (0, 10))
def writeBack(str1, flag, bbox, actionclass, BElist, num, str2):
    producer = KafkaProducer(bootstrap_servers='10.161.178.51:9092', api_version=(1, 0, 0))
    now_time = dt.datetime.now().strftime('%F %T')
    test_info = {
        "car_id_lpn": str1,  # 车辆的车牌 license plate number，识别失败为空字符串
        "car_id_action_flag": flag,
        "car_id_bbox": bbox,  # 车辆位置
        "car_id_action_class": actionclass,  # 私自揽客
        "car_id_start_end_frame": BElist,  # 车辆出现的起止帧号
        "score": num,
        "channel": str2,
        "datatime": now_time,
        "algorithm_name": "szlk_v1",
        "locate": "undetermined",
    }
    # print(test_info)
    msg = json.dumps(test_info).encode()
    # future = producer.send('BUAAshangxianceshi', msg)
    future = producer.send('buaa_cloud', msg)
    result = future.get(timeout=10)
    print(result)
    print(msg)

# if __name__ == "__main__":
# now_time = dt.datetime.now().strftime('%F %T')
# print(now_time)
# writeBack("123ffefe",True,[1,2,3,4],"szlk",[12334,4563423],0.98,"3948049730984")
