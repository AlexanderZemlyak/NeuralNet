// https://rosettacode.org/wiki/Active_Directory/Connect

{$reference System.DirectoryServices.dll}

begin
  var objDE := new System.DirectoryServices.DirectoryEntry
    ('LDAP://DC=onecity,DC=corp,DC=fabrikam,DC=com');
end.