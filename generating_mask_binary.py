#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from xml.dom import minidom
import matplotlib.pyplot as plt
import glob
from skimage.measure import regionprops, label


# In[2]:


from zipfile import ZipFile
#file_name = 'Annotations.zip'
file_name='TissueImages.zip'
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Done') 


# In[3]:


def generate_bit_mask(shape, xml_file):
	"""
	Given the image shape and path to annotations(xml file), 
	generate a bit mask with the region inside a contour being white
	shape: The image shape on which bit mask will be made
	xml_file: path relative to the current working directory 
	where the xml file is present
	Returns: A image of given shape with region inside contour being white..
	"""
	# DOM object created by the minidom parser
	xDoc = minidom.parse(xml_file)

	# List of all Region tags
	regions = xDoc.getElementsByTagName('Region')

	# List which will store the vertices for each region
	xy = []
	for region in regions:
		# Loading all the vertices in the region
		vertices = region.getElementsByTagName('Vertex')

		# The vertices of a region will be stored in a array
		vw = np.zeros((len(vertices), 2))

		for index, vertex in enumerate(vertices):
			# Storing the values of x and y coordinate after conversion
			vw[index][0] = float(vertex.getAttribute('X'))
			vw[index][1] = float(vertex.getAttribute('Y'))
		#x_series = vw[:,0]
		#y_series = vw[:,1]
		#print x_series
		#avg_x=np.mean(x_series)
		#avg_y=np.mean(y_series)
		#print avg_x, avg_y
		#new_coord_x=x_series-avg_x
		#new_coord_y=y_series-avg_y
		#new_coord_x=.5*new_coord_x
		#new_coord_y=.5*new_coord_y
		#new_coord_x=new_coord_x+avg_x
		#new_coord_y=new_coord_y+avg_y

		# print x_series-new_coord_x
		# print y_series - new_coord_y
		#vw[:,0]=new_coord_x
		#vw[:,1]=new_coord_y		


		# Append the vertices of a region
		xy.append(np.int32(vw))

	# Creating a completely black image
	mask = np.zeros(shape, np.uint8)
	# mask for boundaries
	mask_boundary = np.zeros(shape, np.uint8)

	# For each contour, fills the area inside it
	# Warning: If a list of contours is passed, overlapping regions get buggy output
	# Comment out the below line to check, and if the bug is fixed use this
	# cv2.drawContours(mask, xy, -1, (255,255,255), cv2.FILLED)
	for contour in xy:
		cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)
		cv2.drawContours(mask_boundary, [contour], -1, (255,255,255), 2)
	print(np.unique(mask))
	return mask, mask_boundary, len(contour)
#_,mask=generate_bit_mask((1000,1000),'TCGA-CH-5767-01Z-00-DX1.xml')
#mask=cv2.resize(mask,(512,512))
#cv2.imwrite('mask.png',mask)
list_xml = glob.glob('./Annotations/*.xml')
#list_xml = glob.glob('/home/naveen/Desktop/Dataset/input/*.xml')
print(list_xml)
#list_csv = []
for xml in list_xml:
        mask,_,length=generate_bit_mask((1000,1000),xml)
        cv2.imwrite('mask'+xml.split('/')[-1].split('.xml')[0]+'.tif',mask)


# In[ ]:




