##
var f := OpenWrite('a.txt',Encoding.Default);
f.Write('PascalABC.NET по-русски');
f.Close;

f := OpenRead('a.txt',Encoding.GetEncoding(1252));
var s := f.ReadString;
Print(s,s.Length);
f.Close;