import folium
import os
import pandas as pd
import numpy as np

import folium
import utm
from pyproj import Proj

# https://cloud.tencent.com/developer/article/1919827
# 颜色 https://www.cnblogs.com/feffery/p/9282808.html

## An example###########################################################################################
# points=[(31.949515,118.697229), (31.950135,118.696985), (31.950556,118.696913), (31.951091,118.697034), (31.951475,118.697531), (31.951647,118.698275), (31.951669,118.698371)]
# # initiate to plotting area
# my_map = folium.Map(location=[points[0][0],points[-1][1]], zoom_start=12)

# # loop each point
# for i in points:
#     folium.Marker(i).add_to(my_map)
#     #folium.PolyLine(i, color="red", weight=2.5, opacity=1).add_to(my_map)
#     my_map.save("./results/somelocs.html")
#######################################################################################################

#########################UTM 与经纬度互转：#############################################################
# 经纬度转UTM
# lat=37.065
# lon=128.543
# utm_ = utm.from_latlon(lat, lon)
# utm_x = utm_[0]
# utm_y = utm_[1]
# utm_zone = utm_[2]
# utm_band = utm_[3]
# print("utm_x: %s, utm_y: %s, utm_zone: %s, utm_band: %s" % (utm_x, utm_y, utm_zone, utm_band))

# UTM转经纬度
#lat, lon = utm.to_latlon(utm_x, utm_y, utm_zone, utm_band)
########################################################################################################

p = Proj(proj="utm", zone=30, ellps="WGS84")

def read_gps_data(path):
    P = pd.read_csv(path, header=None, dtype=np.double).values  # 读取csv文件，输出为narray
    utm= P[:, 0:2].tolist()  # narray转换成list
    #print("LOCA:", locations_nav)
    locations_nav = utm
    for i in range(len(utm)):
        # zone: 分带号； ellps：参考椭球
        # 默认为北半球 (north=True)，若坐标应在南半球添加south参数，使其为True即可
        lon, lat  = p(utm[i][0], utm[i][1], inverse=True)
        locations_nav[i][0], locations_nav[i][1] = round(lon, 6), round(lat, 6)
        #print("lon,lat", lon, lat)
        #location_nav[i][0], location_nav[i][1] =utm.to_latlon(float(utm[i][0]), float(utm[i][1]))
    return locations_nav

def draw_gps_wave(locations_nav,output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    # 通过修改tiles值来确定底色
    # http://t7.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=
    m = folium.Map(locations_nav[0], zoom_start=30, attr='default', tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}")  # 中心区域的确定
 
    folium.PolyLine(  # polyline方法为将坐标用虚线形式连接起来
        locations_nav,  # 将坐标点连接起来
        weight=2,  # 线的大小为2
        color='blue',  # 线的颜色为蓝色
        opacity=0.8,  # 线的透明度
        dash_array='5'  # 虚线频率
    ).add_to(m)  # 将这条线添加到刚才的区域m内
 
    # 起始点，结束点,QUERY	
    ##residential
    utm=[361205.8468,145137.93]
    location = p(utm[0], utm[1], inverse=True)
    folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
    folium.Circle( radius=100,
                  location=p(362338.982747, 144524.633439, inverse=True), #TOP3
                  color='green', #蓝色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(361061.754142, 145356.48804300002, inverse=True), #TOP2
                  color='#FF66CC', #粉色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(361031.899329, 145061.368643, inverse=True), #TOP1
                  color='crimson', #红色
                  fill=True,
                 ).add_to(m)
    
    # oxford
#     location = locations_nav[430]
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   location=p(5735728.4470069995, 620003.548671, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5734854.257423,  619628.073759, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735747.671261,  619998.342553, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
# #     # U.S.
#     loc= p(363572.174726, 142890.712004,inverse=True)
#     folium.Marker(location=loc, popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363567.539499,  142890.065766, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=50,
#                   location=p(363617.486943,   144314.77665699998, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363559.995847,  144610.65467000002, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
    m.save(os.path.join(output_path, file_name)) 
    
def draw_gps_QPNet(locations_nav,output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    # 通过修改tiles值来确定底色
    # http://t7.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=
    m = folium.Map(locations_nav[0], zoom_start=30, attr='default', tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}")  # 中心区域的确定
 
    folium.PolyLine(  # polyline方法为将坐标用虚线形式连接起来
        locations_nav,  # 将坐标点连接起来
        weight=2,  # 线的大小为2
        color='blue',  # 线的颜色为蓝色
        opacity=0.8,  # 线的透明度
        dash_array='5'  # 虚线频率
    ).add_to(m)  # 将这条线添加到刚才的区域m内
 
    # 起始点，结束点,QUERY	
    # residential
    utm=[361205.8468,145137.93]
    location = p(utm[0], utm[1], inverse=True)
    folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
    folium.Circle( radius=100,
                  location=p(362338.982747, 144524.633439, inverse=True), #TOP3
                  color='green', #蓝色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(361061.754142, 145356.48804300002, inverse=True), #TOP2
                  color='#FF66CC', #粉色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(361031.899329, 145061.368643, inverse=True), #TOP1
                  color='crimson', #红色
                  fill=True,
                 ).add_to(m)
    
    # oxford
#     location = locations_nav[430]
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   #location=p(5735728.4470069995, 620003.548671, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   #location=p(5734854.257423,  619628.073759, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   #location=p(5735747.671261,  619998.342553, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
# #     # U.S.
#     loc= p(363572.174726, 142890.712004,inverse=True)
#     folium.Marker(location=loc, popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363567.539499,  142890.065766, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=50,
#                   location=p(363617.486943,   144314.77665699998, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363559.995847,  144610.65467000002, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)


    
    m.save(os.path.join(output_path, file_name)) 
    
def draw_gps_our(locations_nav,output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    # 通过修改tiles值来确定底色
    # http://t7.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=
    m = folium.Map(locations_nav[0], zoom_start=30, attr='default', tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}")  # 中心区域的确定
 
    folium.PolyLine(  # polyline方法为将坐标用虚线形式连接起来
        locations_nav,  # 将坐标点连接起来
        weight=2,  # 线的大小为2
        color='blue',  # 线的颜色为蓝色
        opacity=0.8,  # 线的透明度
        dash_array='5'  # 虚线频率
    ).add_to(m)  # 将这条线添加到刚才的区域m内
 
    # 起始点，结束点,QUERY	
    ## residential
    # top1: 361031.899329, 'easting': 145061.368643}
    # top2: 361194.823535, 'easting': 145184.656487}
    # top3: 360866.723982, 'easting': 145211.063094}
#     utm=[361205.8468,145137.93]
#     location = p(utm[0], utm[1], inverse=True)
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   location=locations_nav[26], #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[317], #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[11], #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)

    ##ours svt oxford
    utm=[5735557.222185, 620439.36606,]
    location = p(utm[0], utm[1], inverse=True)
    folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   location=p(5735564.50678, 620471.521737, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735564.17555, 620448.223553, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735553.473888, 620431.097327, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    ###svt
    
#     folium.Circle( radius=100,
#                   location=p(5735564.17555, 620448.223553, inverse=True), #TOP1
#                   color='crimson', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735209.785571, 620481.294511, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735176.93001, 620429.176369, inverse=True), #TOP3
#                   color='green',
#                   fill=True,
#                  ).add_to(m)
    
# #     ###ptc
#     folium.Circle( radius=100,
#                   location=p(5735564.17555, 620448.223553, inverse=True), #TOP1
#                   color='crimson', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735230.022617, 620489.973595, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735553.473888, 620431.097327, inverse=True), #TOP3
#                   color='green',
#                   fill=True,
#                  ).add_to(m)

    ###minklocplus
#     folium.Circle( radius=100,
#                   location=p(5735564.17555, 620448.223553, inverse=True), #TOP1
#                   color='crimson', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735553.473888, 620431.097327, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735548.340895, 620412.4582720001, inverse=True), #TOP3
#                   color='green',
#                   fill=True,
#                  ).add_to(m)

#      ###minkloc3dv2 &ppt
#     folium.Circle( radius=100,
#                   location=p(5735553.473888, 620431.097327, inverse=True), #TOP1
#                   color='crimson', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735548.340895, 620412.4582720001, inverse=True), #TOP1
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=p(5735564.17555, 620448.223553, inverse=True), #TOP1
#                   color='green',
#                   fill=True,
#                  ).add_to(m)
    
     ###minkloc
    folium.Circle( radius=100,
                  location=p(5735377.806993, 619963.744512, inverse=True), #TOP1
                  color='crimson', #蓝色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(5735564.17555, 620448.223553, inverse=True), #TOP1
                  color='#FF66CC', #粉色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(5734961.739687, 619718.593496, inverse=True), #TOP1
                  color='green',
                  fill=True,
                 ).add_to(m)
    
#     oxford
#     location = locations_nav[431]
# #     print('location\n', locations_nav[431])
# #     print('location\n', locations_nav[432])
# #     print('location\n', locations_nav[433])
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   location=locations_nav[434], #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[433], #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[432], #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     # U.S.
#     loc= p(363572.174726, 142890.712004,inverse=True)
#     folium.Marker(location=loc, popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363567.539499,  142890.065766, inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=50,
#                   location=p(363827.017518,  143032.05363, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363508.285205,  142944.002218, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)


    
    m.save(os.path.join(output_path, file_name)) 

def draw_gps_org(locations_nav,output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    # 通过修改tiles值来确定底色
    # http://t7.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=
    m = folium.Map(locations_nav[0], zoom_start=30, attr='default', tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}")  # 中心区域的确定
 
    folium.PolyLine(  # polyline方法为将坐标用虚线形式连接起来
        locations_nav,  # 将坐标点连接起来
        weight=2,  # 线的大小为2
        color='blue',  # 线的颜色为蓝色
        opacity=0.8,  # 线的透明度
        dash_array='5'  # 虚线频率
    ).add_to(m)  # 将这条线添加到刚才的区域m内
 
     # 起始点，结束点,QUERY	
#     ##R.A.
    utm=[361205.8468,145137.93]
    location = p(utm[0], utm[1], inverse=True)
    folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
    folium.Circle( radius=100,
                  location=locations_nav[318], #TOP3
                  color='green', #蓝色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=locations_nav[27], #TOP2
                  color='#FF66CC', #粉色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=locations_nav[318], #TOP1
                  color='crimson', #红色
                  fill=True,
                 ).add_to(m)

#     # oxford
#     location = locations_nav[431]
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     folium.Circle( radius=100,
#                   location=locations_nav[435], #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[434], #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[431], #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     # U.S.
#     loc= p(363572.174726, 142890.712004,inverse=True)
#     folium.Marker(location=loc, popup='<b>End Point</b>').add_to(m)
    
#     #folium.Marker(location=locations_nav[390], popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363567.539499,142890.065766,inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=50,
#                   location=p(363559.995847, 144610.65467000002, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363245.327751, 143809.675805, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)


    m.save(os.path.join(output_path, file_name)) 
    
def draw_gps_PCAN(locations_nav,output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    # 通过修改tiles值来确定底色
    # http://t7.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=
    m = folium.Map(locations_nav[0], zoom_start=30, attr='default', tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}")  # 中心区域的确定
 
    folium.PolyLine(  # polyline方法为将坐标用虚线形式连接起来
        locations_nav,  # 将坐标点连接起来
        weight=2,  # 线的大小为2
        color='blue',  # 线的颜色为蓝色
        opacity=0.8,  # 线的透明度
        dash_array='5'  # 虚线频率
    ).add_to(m)  # 将这条线添加到刚才的区域m内
 
     # 起始点，结束点,QUERY	
    ##R.A.
    utm=[361205.8468,145137.93]
    location = p(utm[0], utm[1], inverse=True)
    folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
    folium.Circle( radius=100,
                  location=p(361031.899329, 145061.368643, inverse=True), #TOP3
                  color='green', #蓝色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(361007.356335, 145049.131144, inverse=True), #TOP2
                  color='#FF66CC', #粉色
                  fill=True,
                 ).add_to(m)
    folium.Circle( radius=100,
                  location=p(360866.723982, 145211.063094, inverse=True), #TOP1
                  color='crimson', #红色
                  fill=True,
                 ).add_to(m)

    #oxford
#     location =locations_nav[431]
# #     print('location\n', p(5735747.671261, 619998.342553,inverse=True))
# #     print('location\n', p(5735100.062933, 619869.756772, inverse=True))
# #     print('location\n', p(5735462.6752160005, 619858.958165, inverse=True))
#     folium.Marker(location, popup='<b>End Point</b>').add_to(m)
    
#     #folium.Marker(location=locations_nav[390], popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[432],#location=p(5735747.671261, 619998.342553,inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=100,
#                   location=locations_nav[433], #location=p(5735100.062933, 619869.756772, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=100,
#                   location=locations_nav[434],#location=p(5735462.6752160005, 619858.958165, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)

#     #U.S.
#     loc= p(363572.174726, 142890.712004,inverse=True)
#     folium.Marker(location=loc, popup='<b>End Point</b>').add_to(m)
    
#     #folium.Marker(location=locations_nav[390], popup='<b>End Point</b>').add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363567.539499, 142890.065766,inverse=True), #TOP1
#                   color='crimson', #红色
#                   fill=True,
#                  ).add_to(m)
    
#     folium.Circle( radius=50,
#                   location=p(363552.511443, 144510.3118, inverse=True), #TOP2
#                   color='#FF66CC', #粉色
#                   fill=True,
#                  ).add_to(m)
#     folium.Circle( radius=50,
#                   location=p(363599.97836, 144047.394471, inverse=True), #TOP3
#                   color='green', #蓝色
#                   fill=True,
#                  ).add_to(m)


    m.save(os.path.join(output_path, file_name)) 
    
    
if __name__ == '__main__':
    #path1 = '/userhome/code/pointnetvlad-master/results/RA_run1.csv'  # 前两列预估值，后两列真值，UTM坐标，需要对应到文件当中的.csv重新列表一下；
    path1= '/userhome/code/pointnetvlad-master/results/Oxford_1114163433.csv'
    #path1 = '/userhome/code/pointnetvlad-master/results/US_1.csv'
    name = path1.split('/')[-1].split('.')[0]
    locations_nav= read_gps_data(path1)
    ourname = name +'minkloc.html'
    orgname = name + 'org.html'
    pcanname = name +'pcan.html'
    qname = name +'QPNet.html'
#     svtname = name+'svt.html'
#     minklocname = name+'minkloc.html'
#     minklocplus = name+'minklocplus.html'
#     minkloc3dv2 = name+'minkloc3dv2.html'
#     ppt = name+'ppt.html'
    
#     draw_gps_QPNet(locations_nav, './results/', qname)
    draw_gps_our(locations_nav, './results/', ourname)
    #draw_gps_org(locations_nav, './results/', orgname)
    #draw_gps_PCAN(locations_nav, './results/', pcanname)
    