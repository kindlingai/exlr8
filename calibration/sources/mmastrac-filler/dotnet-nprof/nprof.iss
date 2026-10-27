[Setup]
AppName=NProf
AppVerName=NProf 0.12
AppPublisher=NProf Community
AppPublisherURL=http://code.google.com/p/nprof/
AppSupportURL=http://code.google.com/p/nprof/
AppUpdatesURL=http://code.google.com/p/nprof/
ArchitecturesAllowed=x86 x64
DefaultDirName={pf}/NProf 0.12
DisableProgramGroupPage=yes
OutputDir=Releases
OutputBaseFilename=NProf-0.12-Setup
SetupIconFile=NProf\App.ico
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
UsePreviousAppDir=no

[Languages]
Name: english; MessagesFile: compiler:Default.isl

[Files]
Source: NProf\bin\Release\NProf.exe; DestDir: {app}; Flags: ignoreversion
Source: Hook\Release\ProfilerHook386.dll; DestDir: {app}; Flags: ignoreversion regserver
Source: Hook\x64\Release\ProfilerHookX64.dll; DestDir: {app}; Flags: ignoreversion regserver noregerror

[Icons]
Name: {userprograms}\NProf 0.12; Filename: {app}\NProf.exe
