begin
  var s1: HashSet<string> := HSet('red','green','blue');
  var s2: HashSet<string> := HSet('yellow','magenta','green');
  Println(s1 + s2);
  Println(s1 * s2);
  Println(s1 - s2);
  Println(s1 < s2);
  Println('yellow' in s2);
end.
