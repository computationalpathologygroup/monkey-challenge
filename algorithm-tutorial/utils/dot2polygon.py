import xml.etree.ElementTree as ET

def dot2polygon(xml_path, lymphocyte_half_box_size, monocytes_half_box_size, min_spacing, output_path):
    '''
    :param xml_path (str): the path of the annotation file, ex. root\sub_root\filename.xml
    :param lymphocyte_half_box_size (folat): the size of half of the bbox around the lymphocyte dot in um, 4.5 for lymphocyte
    :param monocytes_half_box_size (folat): the size of half of the bbox around the monocytes dot in um, 11.0 for monocytes
    :param min_spacing (float): the minimum spacing of the wsi corresponding to the annotations
    :param output_path (str): the output path
    :return:
    '''


    # parsing the annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lymphocyte_half_box_size = lymphocyte_half_box_size / min_spacing
    monocytes_half_box_size = monocytes_half_box_size/min_spacing

    # iterating through the dot annotation.
    for A in root.iter('Annotation'):

        #Lymphocytes:
        if (A.get('PartOfGroup')=="lymphocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']
                    sub_child.attrib['X'] = str(float(sub_child.attrib['X'])-lymphocyte_half_box_size)
                    sub_child.attrib['Y'] = str(float(sub_child.attrib['Y'])-lymphocyte_half_box_size)
                child.append(ET.Element(sub_child.tag, Order = '1', X=str(float(x_value)-lymphocyte_half_box_size), Y=str(float(y_value)+lymphocyte_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='2', X=str(float(x_value)+lymphocyte_half_box_size), Y=str(float(y_value)+lymphocyte_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='3', X=str(float(x_value)+lymphocyte_half_box_size), Y=str(float(y_value)-lymphocyte_half_box_size) ))


        # Monoocytes:
        if (A.get('PartOfGroup')=="monocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = sub_child.attrib['X']
                    y_value = sub_child.attrib['Y']
                    sub_child.attrib['X'] = str(float(sub_child.attrib['X'])-monocytes_half_box_size)
                    sub_child.attrib['Y'] = str(float(sub_child.attrib['Y'])-monocytes_half_box_size)
                child.append(ET.Element(sub_child.tag, Order = '1', X=str(float(x_value)-monocytes_half_box_size), Y=str(float(y_value)+monocytes_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='2', X=str(float(x_value)+monocytes_half_box_size), Y=str(float(y_value)+monocytes_half_box_size)))
                child.append(ET.Element(sub_child.tag, Order='3', X=str(float(x_value)+monocytes_half_box_size), Y=str(float(y_value)-monocytes_half_box_size) ))



    # writing the new annotation file
    tree.write(output_path)

