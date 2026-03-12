type
  Person = auto class
  private
    fage: integer;
    procedure SetAge(value: integer);
    begin
      if value < 0 then
        Println('Нечего присваивать отрицательный возраст!')
      else fage := age
    end;
  public
    // свойство - это умное поле
    property Age: integer read fage write SetAge;
  end;
  
begin
  var p := new Person(25);
  Println(p);
  p.Age := -100;
  Println(p);
end.  