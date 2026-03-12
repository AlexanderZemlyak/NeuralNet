// https://rosettacode.org/Globally_replace_text_in_several_files#PascalABC.NET

##
foreach var name in EnumerateFiles('.','*.txt') do
  WriteAllText(name,ReadAllText(name).Replace('Goodbye London!', 'Hello, New York!'));