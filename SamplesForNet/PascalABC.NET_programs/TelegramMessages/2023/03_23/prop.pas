type 
  Person = class
  private
    fname: string;
    fage: integer;
    procedure SetAge(value: integer);
    begin
      if value < 0 then
        value := 0;
      fage := value;
    end;
  public
    constructor(name: string; age:integer) := (fname,fage) := (name,age);
    property Name: string read fname;
    property Age: integer read fage write SetAge;
  end;
  
begin
  var p := new Person('Иванов',17);
  Println(p);
  p.Age := -5;
  Println(p);
end.  