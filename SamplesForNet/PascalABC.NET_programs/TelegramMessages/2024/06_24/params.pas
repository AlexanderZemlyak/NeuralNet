procedure ppp(params a: array of object);
begin
  Println(a.Count);
end;

begin
  var a: array of object := (1,2,3);
  ppp(a);
  ppp(a,a);
  var ai: array of integer := (1,2,3);
  ppp(ai);
end.