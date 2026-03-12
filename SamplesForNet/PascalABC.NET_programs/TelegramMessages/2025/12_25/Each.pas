begin  
  var s := 'как однажды жак звонарь головой сломал фонарь';
  s.ToWords
   .GroupBy(word -> word.Length)
   .Each(word -> word.Count)
   .OrderBy(kv -> kv.Key)
   .PrintLines(kv -> $'{kv.Key}-буквенных - {kv.Value} шт')
end.
