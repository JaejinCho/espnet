# Program to demonstrate 'Variable 
# defined inside the class' 

# print("Inside_class1", inVar) # Error 

''' Class one'''
class Geek: 

	# Variable defined inside the class. 
	inVar = 'inside_class'
	print("Inside_class2", inVar) 

	def access_method(self): 
		print("Inside_class3", self.inVar) 

uac = Geek() 
uac.access_method() 

''' Class two '''
class another_Geek_class: 
	print() 
# print("Inside_class4", inVar) # Error 

	def another_access_method(self): 
		print() 
# print("Inside_class5", inVar) # Error 

uaac = another_Geek_class() 
uaac.another_access_method() 

