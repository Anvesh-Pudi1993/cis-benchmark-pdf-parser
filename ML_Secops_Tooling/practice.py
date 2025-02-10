class welcome:
    def __init__(self,name):
        self.name=name
    def greet(self):
        return f"Hello ,{self.name}!"
def main():
    obj=welcome("Anvesh")
    print(obj.greet())

if __name__=='main':
    main()
    