begin
  var obj: object := 'Hello PascalABC.NET!';
  match obj with
    integer(i): Println(i * i);
    string(s): Println(s.Length);
    else Println(TypeName(obj));
  end;
end.