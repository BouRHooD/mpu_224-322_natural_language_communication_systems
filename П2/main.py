import  jpype     
import  asposecells     
jpype.startJVM() 
from asposecells.api import Workbook
workbook = Workbook("СборникСтатейДляОбучения.txt")
workbook.save("СборникСтатейДляОбучения.tsv")
jpype.shutdownJVM()
	