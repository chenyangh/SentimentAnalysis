import re
from lxml import etree
from bz2 import BZ2File
from sets.core import Step, Dataset


class Wikipedia(Step):
    """
    Titles and plain text of Wikipedia articles. By default, this is extracted
    from the English dump of May 2016. For an overview of available Wikipedia
    dumps visit: https://dumps.wikimedia.org/backup-index.html
    """

    TOKEN_REGEX = re.compile(r'[A-Z]*[a-z]+|[A-Z]+[a-z]*')

    def __new__(cls, url='https://dumps.wikimedia.org/enwiki/20160501/'
                'enwiki-20160501-pages-meta-current.xml.bz2', amount=None):
        filepath = cls.download(url)
        dataset = cls.disk_cache('dataset', cls._parse, filepath, amount)
        return dataset

    @classmethod
    def _parse(cls, filepath, amount=None):
        pages = []
        with BZ2File(filepath) as file_:
            for element in cls._stream(file_, '{*}page'):
                if amount and len(pages) >= amount:
                    break
                page = cls._process(element)
                if page:
                    pages.append(page)
        ids, titles, contents = zip(*pages)
        return Dataset(ids=ids, title=titles, content=contents)

    @classmethod
    def _process(cls, element):
        if element.find('./{*}redirect') is not None:
            return
        id_ = int(element.findtext('./{*}id'))
        title = cls._extract(element.findtext('./{*}title'))
        text = cls._extract(element.findtext('./{*}revision/{*}text'))
        return id_, title, text

    @classmethod
    def _extract(cls, string):
        if not string:
            return ''
        tokens = cls.TOKEN_REGEX.findall(string)
        tokens = [x.lower() for x in tokens]
        text = ' '.join(tokens)
        return text

    @classmethod
    def _stream(cls, file_, tag):
        context = etree.iterparse(file_, tag=tag, events=['end'])
        for _, element in context:
            yield element
            element.clear()
            for ancestor in element.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        del context
