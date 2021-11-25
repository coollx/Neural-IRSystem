import xml.etree.ElementTree as ET
import os

class QueryParser:
    sep = os.path.sep
    
    def parse(self, query_file_path = 'files' + sep + 'topics_MB1-49.txt'):
        with open(query_file_path, 'r', encoding = 'utf-8') as f1:
            with open('files' + self.sep + 'query.xml', 'w', encoding = 'utf-8') as f2:
                f2.write('<root>\n')
                f2.write(f1.read())
                f2.write('\n</root>')
        root = ET.parse('files' + self.sep + 'query.xml').getroot()
        ret = list()
        for query in root:
            ret.append(query.find('title').text)
        return ret


if __name__ == '__main__':
    qp = QueryParser()
    print(qp.parse())