begin
  var text := ReadAllText('AllDelimiters.pas');
  text.ToWords(AllDelimiters).PrintLines
end.