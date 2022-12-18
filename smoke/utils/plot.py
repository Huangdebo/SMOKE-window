# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:00:06 2022

@author: hdb
"""

def plot_box(draw, box2d=[], box3d=[]):

    if len(box2d) > 0:
        num_obj = box2d.shape[0]
        for i in range(num_obj):  
    
            x1, y1, x2, y2 = box2d[i].split(1, 0)
            draw.rectangle([x1,y1,x2,y2], outline=(0, 255, 0))
        
    if len(box3d) > 0:
        box3d = box3d.permute(0, 2, 1)
        num_obj = box3d.shape[0]
        for i in range(num_obj):  

            # for j in range(box3d[i].shape[0]):
            #     x, y = box3d[i][j].split(1, 0)
            #     draw.ellipse((x-2,y-2, x+2,y+2), (255,0,0)) 

            point_pair = [
                [box3d[i][0].split(1, 0), box3d[i][1].split(1, 0)],
                [box3d[i][0].split(1, 0), box3d[i][5].split(1, 0)],
                [box3d[i][0].split(1, 0), box3d[i][7].split(1, 0)],
                [box3d[i][2].split(1, 0), box3d[i][1].split(1, 0)],
                [box3d[i][2].split(1, 0), box3d[i][3].split(1, 0)],
                [box3d[i][2].split(1, 0), box3d[i][7].split(1, 0)],
                [box3d[i][4].split(1, 0), box3d[i][1].split(1, 0)],
                [box3d[i][4].split(1, 0), box3d[i][3].split(1, 0)],
                [box3d[i][4].split(1, 0), box3d[i][5].split(1, 0)],
                [box3d[i][6].split(1, 0), box3d[i][3].split(1, 0)],
                [box3d[i][6].split(1, 0), box3d[i][5].split(1, 0)],
                [box3d[i][6].split(1, 0), box3d[i][7].split(1, 0)]
                ]

            for j in range(len(point_pair)):
                draw.line(point_pair[j], (0,255,0), width=2) 
            
            
            
            
            
            