type
  Person = auto class
    name: string;
    age: integer;
  end;
  Student = class(Person)
  public
    univ: string;
    constructor(n: string; a: integer; u: string);
    begin
      inherited Create(n,a);
      univ := u;
    end;  
  end;

begin
  var L := new List<Student>;
  L.Add(new Student('Иванов',20,'ЮФУ'));
  L.Add(new Student('Попова',19,'ЮФУ'));
  var ip: sequence of Person := L;
  ip.Print
end.