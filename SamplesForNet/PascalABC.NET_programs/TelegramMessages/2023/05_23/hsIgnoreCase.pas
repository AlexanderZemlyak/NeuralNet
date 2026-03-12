begin
  var hs := new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
  hs.Add('hello');
  hs.Add('Hello');
  hs.Add('GoodBye');
  hs.Add('goodbye');
  hs.Println;
  Print('HELLO' in hs);
end.