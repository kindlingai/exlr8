/***************************************************************************
                          Hook.cpp  -  description
                             -------------------
    begin                : Sat Jan 18 2003
    copyright            : (C) 2003, 2004, 2005, 2006, 2007, 2008, 2009 by Matthew Mastracci, Christian Staudenmeyer
    email                : staudenmeyer@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma warning ( disable : 4530 )
#include <atlbase.h>
#include <atlcom.h>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include "cor.h"
#include "corprof.h"
#include "Dbghelp.h"
#include "resource.h"
#include "CorHdr.h"
#include "winbase.h"

using namespace ATL;
using namespace std;

const int MAX_FUNCTION_LENGTH=2048;
#define guid "61D3E5D7-53E9-46B2-8DBC-011099A91793"


namespace ProfilerHook
{
	class FunctionInfo 
	{
	public:
		FunctionInfo(string signature, string nameSpace)
		{
			this->signature=signature;
			this->nameSpace=nameSpace;
		}
		string signature;
		string nameSpace;
	};
	class ProfilerHelper
	{
	public:
		// TODO: remove this
		string GetClass(string className,string& nameSpace) 
		{
			int index=className.length();
			for(;index>0 && className[index-1]!='.';index--);
			nameSpace=className.substr(0,index-1);
			return className.substr(index);
		}
		// TODO: use strings
		HRESULT GetClassName(IMetaDataImport* pIMetaDataImport, mdTypeDef tkType, wchar_t* szClass, wchar_t* szParentClass, wchar_t* szNamespace)
		{
		   HRESULT hr = S_OK;

		   wchar_t szFullClassName[MAX_CLASS_NAME+1];
		   szFullClassName[MAX_CLASS_NAME] = L'\0';

		   ULONG Length = 0;
		   CorTypeAttr tdFlags = tdClass;   // 0 value
		   mdToken tkParent = 0;
		   hr = pIMetaDataImport->GetTypeDefProps(
				 tkType,
				 szFullClassName, 
				 MAX_CLASS_NAME,
				 &Length,
				 (DWORD*)&tdFlags, 
				 &tkParent
				 );
		         
		   // This could be S_FALSE in some cases.
		   //ASSERT(SUCCEEDED(hr));
		   if (FAILED(hr)) return(hr);

		   // The full name is retrieved. If it's a nested type, szFullClassName 
		   // is the class name only, and I need to do the same for its parent.
		   // Otherwise, it contains the namespace plus the class name at the end.
		   if (!IsTdNested(tdFlags))
		   {
			  // Not a nested type: 
			  // --> szFullClassName = namespace.namespace...namespace.ClassName
			  wchar_t* pszNamespace = szFullClassName;
			  wchar_t* pszDot = wcsrchr(szFullClassName, L'.');
			  if (pszDot == NULL)
			  {
				 // No namespace, just the class name
				 szNamespace[0] = L'\0';
				 wcsncpy(szClass, szFullClassName, MAX_CLASS_NAME);
			  }
			  else
			  {
				 // The '.' is just between the namespaces and the class name.
				 // The namespace itself starts after the '.'.
				 *pszDot = L'\0';
				 wchar_t* pszName = pszDot + 1;

				 // Extract class and namespace         
				 wcsncpy(szClass, pszName, MAX_CLASS_NAME);
				 wcsncpy(szNamespace, pszNamespace, MAX_CLASS_NAME);
			  }

			  // and no parent class
			  szParentClass[0] = L'\0';
		   }
		   else
		   {
			  // Get the token for its parent
			  hr = pIMetaDataImport->GetNestedClassProps(tkType, &tkParent);
			  //ASSERT(SUCCEEDED(hr));
		   
			  // szFullClassName is the class name and we have 
			  // to search recursively starting from its parent.
			  wchar_t szTmpClass[MAX_CLASS_NAME+1];
			  szTmpClass[MAX_CLASS_NAME] = L'\0';
			  wchar_t szTmpParentClass[MAX_CLASS_NAME+1];
			  szTmpParentClass[MAX_CLASS_NAME] = L'\0';

			  hr = GetClassName(pIMetaDataImport, tkParent, 
				szTmpClass, szTmpParentClass, szNamespace);
			  //ASSERT(SUCCEEDED(hr));
		      
			  // build the class name from its parent name
			  if (szTmpParentClass[0] != L'\0')
			  {
				 size_t length = 0;

				 wcsncpy(szParentClass, szTmpParentClass, MAX_CLASS_NAME);
				 length = wcslen(szTmpParentClass); // count string just copied

				 wcsncat(szParentClass, L".", MAX_CLASS_NAME - length);
				 length++; // count the '.'

				 wcsncat(szParentClass, szTmpClass, MAX_CLASS_NAME - length);
			  }
			  else wcsncpy(szParentClass, szTmpClass, MAX_CLASS_NAME);
		         
			  // The single class name is returned by GetTypeDefProps()
			  wcsncpy(szClass, szFullClassName, MAX_CLASS_NAME);

			  // Nothing special to do for namespace as it was already
			  // computed during the parent lookup.
		   }
		   return(hr);
		}

		// TODO: refactor
		FunctionInfo* GetFunctionSignature(FunctionID functionId) 
		{
			FunctionInfo* function=new FunctionInfo("","");
			
			if(functionId==17) {
				function->signature="unknown stack frame";
				function->nameSpace="Unknown stack frames";
				return function;
			}		
			string name;
			ClassID classID;
			ModuleID moduleID;
			mdToken moduleToken;
			if(FAILED(profilerInfo->GetFunctionInfo(functionId,&classID,&moduleID,&moduleToken))) 
			{
				// TODO: throw exceptions here
				DebugBreak();
			}

			IMetaDataImport *metaDataImport = 0;	
			mdToken	token;
			// TODO: fix deadlock
			if(FAILED(profilerInfo->GetTokenAndMetaDataFromFunction(functionId, IID_IMetaDataImport, (IUnknown **)&metaDataImport,&token))) 
			{
				//DebugBreak();
			}

			WCHAR functionName[MAX_FUNCTION_LENGTH];
			if(FAILED(metaDataImport->GetMethodProps(token, 0, functionName, MAX_FUNCTION_LENGTH,0,0,0,0,0,0))) 
			{
				//DebugBreak();
			}

			mdTypeDef classToken = 0;
			profilerInfo->GetClassIDInfo(classID, 0, &classToken);
			if(classToken != mdTypeDefNil) {
				
				WCHAR className[MAX_CLASS_NAME+1];
				WCHAR parentClass[MAX_CLASS_NAME+1];
				WCHAR classNamespace[MAX_CLASS_NAME+1];
				GetClassName(metaDataImport,classToken,className,parentClass,classNamespace);

				if(parentClass[0]!='\0')
				{
					name+=(string)CW2A(parentClass)+".";
				}
				if(className[0]!='\0')
				{
					name+=(string)CW2A(className)+".";
				}
				name+=(string)CW2A(functionName);
				function->nameSpace=(string)CW2A(classNamespace);
				if(function->nameSpace.empty())
				{
					function->nameSpace="No namespace";
				}
			}
			else
			{
				function->nameSpace="No namespace";
			}
			DWORD methodAttr = 0;
			PCCOR_SIGNATURE sigBlob = 0;
			if(FAILED(metaDataImport->GetMethodProps( (mdMethodDef) token,0,0,0,0,&methodAttr,&sigBlob,0,0,0 ))) 
			{
				DebugBreak();
			}

			ULONG callConv;
			sigBlob += CorSigUncompressData(sigBlob, &callConv);
			name+="(";

			ULONG argCount=0;
			sigBlob += CorSigUncompressData(sigBlob, &argCount);
			sigBlob--;
			string returnType=ParseElementType(metaDataImport, &sigBlob);
			if(sigBlob!=0) 
			{
				for(ULONG i = 0; (sigBlob != 0) && (i < (argCount)); i++) 
				{
					if(i!=0)
					{
						name+=", ";
					}
					name+=ParseElementType(metaDataImport,&sigBlob);
				}
			}

			name+=")";
			metaDataImport->Release();
			function->signature=name;
			return function;
		}
		string ParseElementType(IMetaDataImport *metaDataImport,PCCOR_SIGNATURE* ppSignature) 
		{
			ULONG count=0;
			(*ppSignature)++;
			CorElementType type=(CorElementType)**ppSignature;
			switch((CorElementType)type) {
				case ELEMENT_TYPE_VOID: return "void";
				case ELEMENT_TYPE_BOOLEAN: return "bool";
				case ELEMENT_TYPE_CHAR: return "wchar";
				case ELEMENT_TYPE_I1: return "byte";
				case ELEMENT_TYPE_U1: return "ubyte";
				case ELEMENT_TYPE_I2: return "short";
				case ELEMENT_TYPE_U2: return "ushort";
				case ELEMENT_TYPE_I4: return "int";
				case ELEMENT_TYPE_U4: return "uint";
				case ELEMENT_TYPE_I8: return "long";
				case ELEMENT_TYPE_U8: return "ulong";
				case ELEMENT_TYPE_R4: return "float";
				case ELEMENT_TYPE_R8: return "double";
				case ELEMENT_TYPE_U: return "uint";
				case ELEMENT_TYPE_I: return "int";
				case ELEMENT_TYPE_OBJECT: return "object";
				case ELEMENT_TYPE_STRING: return "string";
				case ELEMENT_TYPE_VAR :  
				  {  
					return "var";
				  }  
				  break;  
				case ELEMENT_TYPE_TYPEDBYREF: return "refany";
				case ELEMENT_TYPE_CLASS: 
				case ELEMENT_TYPE_VALUETYPE:
				case ELEMENT_TYPE_CMOD_REQD:
				case ELEMENT_TYPE_CMOD_OPT:
					mdToken	token;
					(*ppSignature)++;
					(*ppSignature) += CorSigUncompressToken((PCCOR_SIGNATURE)(*ppSignature),&token);
					if(TypeFromToken(token)!=mdtTypeRef) {
						WCHAR zName[MAX_FUNCTION_LENGTH];
						metaDataImport->GetTypeDefProps(token,zName,MAX_FUNCTION_LENGTH,0,0,0);
						(*ppSignature)--;
						string nameSpace;
						return GetClass((string)CW2A(zName),nameSpace);
					}
					return "mdtTypeRef";
				case ELEMENT_TYPE_SZARRAY:
					return ParseElementType(metaDataImport,ppSignature)+"[]";
				case ELEMENT_TYPE_ARRAY: {	
					ULONG rank;
					string text=ParseElementType(metaDataImport,ppSignature);
					rank = CorSigUncompressData((PCCOR_SIGNATURE&)*ppSignature);
					if(rank == 0) 
					{
						text+="[?]";
					}
					else 
					{
						ULONG *lower;
						ULONG *sizes;
						ULONG numsizes;
						ULONG arraysize = (sizeof ( ULONG ) * 2 * rank);
		                
						lower=(ULONG*)_alloca(arraysize);
						memset(lower,0,arraysize);
						sizes = &lower[rank];

						numsizes = CorSigUncompressData((PCCOR_SIGNATURE&)*ppSignature);
						if(numsizes <= rank)
						{
							ULONG numlower;			                    
							for(ULONG i = 0;i<numsizes;i++)
							{
								sizes[i]=CorSigUncompressData((PCCOR_SIGNATURE&)*ppSignature);
							}			                    
							numlower = CorSigUncompressData((PCCOR_SIGNATURE&)*ppSignature);	
							if(numlower <= rank)
							{
								for(ULONG i = 0;i < numlower;i++)
								{
									lower[i] = CorSigUncompressData((PCCOR_SIGNATURE&)*ppSignature);
								}
								text+="[";
								for (ULONG i = 0; i < rank; i++ )
								{	
									if ( (sizes[i] != 0) && (lower[i] != 0) )
									{
										if ( lower[i] == 0 )
										{
											text+=ToString(sizes[i]);
										}
										else
										{
											text+=ToString(lower[i]);	
											text+="...";
											if(sizes[i] != 0)
											{
												text+=ToString((lower[i] + sizes[i] + 1));
											}
										}
									}
									if (i < (rank - 1))
									{
										text+=",";
									}
								}          
								text+="]";
							}
						}
					}
					return text;
				}
				case ELEMENT_TYPE_GENERICINST: {
					//string text="generic type: ";
					//text+=ParseElementType(metaDataImport,ppSignature);
					////ParseElementType(
					////(*ppSignature) += CorSigUncompressToken((PCCOR_SIGNATURE)(*ppSignature),&token);
					//////(*ppSignature)--;
					//////if(TypeFromToken(token)!=mdtTypeRef) {
					////	WCHAR zName[MAX_FUNCTION_LENGTH];
					////	metaDataImport->GetTypeDefProps(token,zName,MAX_FUNCTION_LENGTH,0,0,0);
					////	text+=(string)CW2A(zName);
					//////}
					//ULONG count=0;
					//(*ppSignature) += CorSigUncompressData(*ppSignature, &count);
					//text+="<";
					//text+=ToString(count);
					//return text;
					return "generic type";
				}
				case ELEMENT_TYPE_PINNED:
					return ParseElementType(metaDataImport,ppSignature)+"pinned";
				case ELEMENT_TYPE_PTR:   
					return ParseElementType(metaDataImport,ppSignature)+"*";
				case ELEMENT_TYPE_BYREF:   
					return ParseElementType(metaDataImport,ppSignature)+"&";
				case ELEMENT_TYPE_END:
					//return "<end>";
				default:
					return "<UNKNOWN>"+ToString(type);
			}
		}
		template<class T> string ToString(T i) 
		{
			string s;
			stringstream stream;
			stream<<i;
			stream>>s;
			return s;
		}
		CComPtr<ICorProfilerInfo2> profilerInfo;
	};
	HRESULT __stdcall __stdcall ManagedFrameCallback(
		FunctionID funcId,
		UINT_PTR ip,
		COR_PRF_FRAME_INFO frameInfo,
		ULONG32 contextSize,
		BYTE context[],
		void *clientData
	)
	{
		if(funcId!=0) 
		{
			((vector<FunctionID>*)clientData)->push_back(funcId);
		}
		return S_OK;
	}
	const int interval=5;
	class Profiler 
	{
		public: 
		vector<vector<FunctionID>*> stackWalks;
		ofstream* file;

		map< FunctionID, FunctionInfo*> signatures;
		
		string GetTemporaryFileName() {
			char path[MAX_PATH];
			path[0]=0;
			GetTempPath(MAX_PATH-1,path);
			return (string)path+guid+".nprof";
		}
		void EndAll(ProfilerHelper& profilerHelper) {
			//cout << "ending"<<endl;
			//string s;
			//cin >> s;
			file=new ofstream(GetTemporaryFileName().c_str(), ios::binary);
			for(map< FunctionID, FunctionInfo*>::iterator i=signatures.begin();i!=signatures.end();i++) {
				WriteInteger(i->first);
				WriteString(i->second->signature);
				WriteString(i->second->nameSpace);
			}
			WriteInteger((unsigned __int64)-1);
			for(vector<vector<FunctionID>*>::iterator stackWalk = stackWalks.begin(); stackWalk != stackWalks.end(); stackWalk++ ) {
				for(vector<FunctionID>::iterator stackFrame=(*stackWalk)->begin();stackFrame!=(*stackWalk)->end();stackFrame++) {
					WriteInteger(*stackFrame);
				}
				WriteInteger((unsigned __int64)-1);
			}
			WriteInteger((unsigned __int64)-1);
			WriteInteger((unsigned __int64)-2);
			file->close();
			delete file;
		}
		void WriteString(const string& signature) {
			WriteInteger((unsigned __int64)signature.length());
			file->write(signature.c_str(),(int)signature.length());
		}
		void WriteInteger(unsigned __int64 id) {
			file->write((char*)&id,sizeof(unsigned __int64));
		}
		int count;
		Profiler::Profiler(ICorProfilerInfo2* profilerInfo) {
			InitializeCriticalSection(&threadMapLock);
			this->profilerInfo = profilerInfo;
			this->profilerHelper.profilerInfo=profilerInfo;
			profilerInfo->SetEventMask(COR_PRF_ENABLE_STACK_SNAPSHOT|COR_PRF_MONITOR_THREADS|COR_PRF_DISABLE_INLINING|COR_PRF_MONITOR_GC);
			process=GetCurrentProcess();
			count=0;
			if (!SymInitialize(process, NULL, TRUE))
			{
				cout << "SymInitialize failed";
			}
			//isVista=false;
			HMODULE kernel32 = LoadLibraryW(L"Kernel32");
			if (kernel32 != NULL)
			{    
				FARPROC proc=GetProcAddress(kernel32, "QueryThreadCycleTime");
				if (proc != NULL)    
				{        
					//isVista=true;
				}
			}
			//DebugBreak();
			SetTimer();
		}
		HANDLE process;
		void KillTimer() {
			timeKillEvent(timer);
		}
		void SetTimer() {
			TIMECAPS timeCaps;
			timeGetDevCaps(&timeCaps, sizeof(TIMECAPS));
			timer = timeSetEvent(interval,timeCaps.wPeriodMin,(LPTIMECALLBACK)TimerFunction,	(DWORD_PTR)this,TIME_PERIODIC);
		}
		// Called by the profiler hook when a managed thread is assigned to an OS thread
		void ThreadAssigned( ThreadID managedThreadId, DWORD dwOSThread )
		{
			EnterCriticalSection(&threadMapLock);
			threadMap[dwOSThread]=managedThreadId;

			vistaThreadTime[managedThreadId]=0;

			FILETIME time;
			time.dwHighDateTime=0;
			time.dwLowDateTime=0;
			threadTime[managedThreadId]=time;

			LeaveCriticalSection(&threadMapLock);
		};
		
		// Called by the profiler hook when a managed thread is shut down.
		// At this stage, we must stop stack walking it or 'bad things happen'
		void ThreadDestroyed( ThreadID managedThreadId )
		{
			EnterCriticalSection(&threadMapLock);
			
			for(map< DWORD, ThreadID >::iterator it=threadMap.begin();it!=threadMap.end();it++)
			{
				ThreadID itManagedThreadId=(*it).second;
				if (itManagedThreadId == managedThreadId)
				{
					threadMap.erase(it);
					break;
				}
			}		
			LeaveCriticalSection(&threadMapLock);
		};

		void ThreadMap(ThreadID threadId, DWORD dwOSThread) {
			threadMap[dwOSThread] = threadId;
		}
		virtual void End() {
			KillTimer();
			EndAll(profilerHelper);
		}

		static void CALLBACK TimerFunction(UINT wTimerID, UINT msg, DWORD dwUser, DWORD dw1, DWORD dw2) {
			Profiler* profiler=(Profiler*)dwUser;
			profiler->KillTimer();
			// ignore first callback to avoid deadlock
			if(profiler->count>1)
			{
				profiler->WalkStack();
			}
			profiler->count++;
			profiler->SetTimer();
		}
		static UINT timer;
		map<DWORD,DWORD> switchMap;
		bool DecideWalk(HANDLE threadHandle,ThreadID id)
		{
			//if(isVista)
			//{
				ULONG64 cycles=0;
				QueryThreadCycleTime(threadHandle,&cycles);
				ULONG64 t=vistaThreadTime[id];
				//DebugBreak();

				int diff=3000000;
				if( cycles - t > diff)
				{
					//cout << cycles -t <<endl;
					//vistaThreadTime[id]=t+diff;
					vistaThreadTime[id]=cycles;
					//vistaThreadTime[id]=cycles;
					return true;
				}
				else
				{
					//cout << "not walking" << endl;
					return false;
				}
			//}
			//else
			//{
			//	FILETIME creationTime;
			//	FILETIME exitTime;
			//	FILETIME kernelTime;
			//	FILETIME userTime;
			//	FILETIME t=threadTime[id];
			//	GetThreadTimes(
			//	  threadHandle,
			//	  &creationTime,
			//	  &exitTime,
			//	  &kernelTime,
			//	  &userTime
			//	);

			//	if(CompareFileTime(&userTime,&t)>0)
			//	{
			//		threadTime[id]=userTime;
			//		return true;
			//	}
			//	else
			//	{
			//		return false;
			//	}

			//}
		}
		//bool isVista;
		void WalkStack() {
			EnterCriticalSection(&threadMapLock);
			for(map<DWORD,ThreadID>::iterator i=threadMap.begin();i!=threadMap.end();i++) {
				DWORD threadId=i->first;
				HANDLE threadHandle=OpenThread(THREAD_SUSPEND_RESUME|THREAD_QUERY_INFORMATION|THREAD_GET_CONTEXT,false,threadId);
				int suspendCount=SuspendThread(threadHandle);
				CloseHandle(threadHandle);
			}
			//if(stackWalks.size()==0)
			//{
			//	DebugBreak();
			//}

			for(map<DWORD,ThreadID>::iterator i=threadMap.begin();i!=threadMap.end();i++) {

				DWORD threadId=i->first;
				//cout << i->first<< endl;
				//cout << i->second << endl;

				HANDLE threadHandle=OpenThread(THREAD_SUSPEND_RESUME|THREAD_QUERY_INFORMATION|THREAD_GET_CONTEXT,false,threadId);
				//cout << threadHandle <<endl;

				ThreadID id=threadMap[threadId];

				//cout << id << endl;

				if(DecideWalk(threadHandle,id))
				{
					//cout << "walking thread " << id << endl;
					
					vector<FunctionID>* functions=new vector<FunctionID>();
					CONTEXT context;
					context.ContextFlags=CONTEXT_FULL;
					GetThreadContext(threadHandle,&context);

					//while(true)
					//{
						// TODO: check result, combine with below
						profilerInfo->DoStackSnapshot(
							id,
							ManagedFrameCallback,
							COR_PRF_SNAPSHOT_DEFAULT,
							functions,
							(BYTE*)&context,sizeof(context)
						);
						//if(functions->size()!=0)
						//{
						//	//cout << "thread id" << threadId<<endl;
						//	//string s;
						//	//cin >>s;
						//	//cout << "tries: 0" << endl;
						//	break;
						//	/*break;*/
						//}

						if(functions->size()==0)
						{
							//cout << "thread id" << threadId<<endl;
							//string s;
							//cin >>s;
							//cout << "tries: 0" << endl;
							/*break;*/


						STACKFRAME64 sf64;
						memset(&sf64,0,sizeof(STACKFRAME64));
						#ifdef _M_IX86
						sf64.AddrPC.Offset=context.Eip;
						sf64.AddrStack.Offset=context.Esp;
						sf64.AddrFrame.Offset=context.Ebp;
						#elif _M_X64
						sf64.AddrPC.Offset=context.Rip;
						sf64.AddrStack.Offset=context.Rsp;
						sf64.AddrFrame.Offset=context.Rsp;
						#endif
						sf64.AddrPC.Mode=AddrModeFlat;
						sf64.AddrStack.Mode=AddrModeFlat;
						sf64.AddrFrame.Mode=AddrModeFlat;

						FunctionID functionId=0;
						int count=0;
						while(StackWalk64(
							#ifdef _M_IX86
							IMAGE_FILE_MACHINE_I386,
							#elif _M_X64
							IMAGE_FILE_MACHINE_AMD64,
							#endif
							GetCurrentProcess(),
							GetCurrentThread(),
							&sf64,
							&context,
							0,
							SymFunctionTableAccess64,
							SymGetModuleBase64,
							0
						))
						{
							WORD  dwAddress;
							DWORD  dwDisplacement;
							ULONG64 buffer[(sizeof(SYMBOL_INFO) +MAX_SYM_NAME * sizeof(TCHAR) +
								sizeof(ULONG64) - 1) /sizeof(ULONG64)];
							
							PSYMBOL_INFO pSymbol = (PSYMBOL_INFO) buffer;
							pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
							pSymbol->MaxNameLen = MAX_SYM_NAME;

							string symbol;
							FunctionID f;
							if (SymFromAddr(process, (DWORD64)sf64.AddrPC.Offset, (PDWORD64)&dwDisplacement, pSymbol))
							{
								symbol=pSymbol->Name;
								f=pSymbol->Address;
							}
							else
							{
								std::string s;
								std::stringstream out;
								out << "0x" << setbase(16) << sf64.AddrPC.Offset;
								symbol = out.str();
								symbol = "Unknown native function";
								f=sf64.AddrPC.Offset;
							}
							// todo: is this correct?
							if(sf64.AddrPC.Offset==0)
							{
								break;
							}
							//IMAGEHLP_MODULE module;
							//if(SymGetModuleInfo(process,sf64.AddrPC.Offset,&module))
							//{
							//	symbol=symbol+ "(" + module.ModuleName + ")";
							//}
							/*FunctionID f=sf64.AddrPC.Offset;*/
							functions->push_back(f);

							const map< FunctionID, FunctionInfo* >::iterator found = signatures.find(f);
							if(found == signatures.end()){
								FunctionInfo* function=new FunctionInfo(symbol,"Native code");
								signatures.insert(make_pair(f,function));
							}

							int x=functions->size();

							FunctionID a=0;
							CONTEXT c=context;
							// TODO: check result
							profilerInfo->DoStackSnapshot(
								id,
								ManagedFrameCallback,
								COR_PRF_SNAPSHOT_DEFAULT,
								functions,
								(BYTE*)&c,sizeof(c)
							);
							if(functions->size()!=x)
							{
								//cout << "thread id" << threadId<< " with native " << endl;
								//string s;
								//cin >>s;
								break;
							}
							if(x>500)
							{
								break;
							}
							count++;
						}
						//cout << "tries: " << count << endl;

						if(functions->size()==0)
						{
							functions->push_back(17);
						//	break;
						}
						//else
						//{							
						//	break;
						//}
						cout << "several" << endl;
					}
					//}
					for(int index=0;index<functions->size();index++) {
						FunctionID id=functions->at(index);
						const map< FunctionID, FunctionInfo* >::iterator found = signatures.find(id);
						if(found == signatures.end()){
							FoundNewFunction(id);
						}
					}
					stackWalks.push_back(functions);
				}
				CloseHandle(threadHandle);
			}
			for(map<DWORD,ThreadID>::iterator i=threadMap.begin();i!=threadMap.end();i++) {
				DWORD threadId=i->first;
				HANDLE threadHandle=OpenThread(THREAD_SUSPEND_RESUME|THREAD_QUERY_INFORMATION|THREAD_GET_CONTEXT,false,threadId);
				if(ResumeThread(threadHandle)>1)
				{
					DebugBreak();
				}
				CloseHandle(threadHandle);
			}
			LeaveCriticalSection(&threadMapLock);
		}
		void FoundNewFunction(FunctionID functionId)
		{
			FunctionInfo* function = profilerHelper.GetFunctionSignature(functionId);
			signatures.insert(make_pair(functionId,function));
		}
		CRITICAL_SECTION threadMapLock;
		map<DWORD,ThreadID> threadMap;
		map<ThreadID,ULONG64> vistaThreadTime;
		map<ThreadID,FILETIME> threadTime;
	protected:
		CComPtr<ICorProfilerInfo2> profilerInfo;
		ProfilerHelper profilerHelper;
	};
	UINT Profiler::timer;
	[
		object,
		uuid("FDEDE932-9F80-4CE5-891E-3B24768CFBCB"),
		dual,helpstring("INProfCORHook Interface"),
		pointer_default(unique)
	]
	__interface INProfCORHook : IDispatch 
	{
	};
	[
	  coclass,
	  threading("apartment"),
	  vi_progid("NProf.NProfCORHook"),
	  progid("NProf.NProfCORHook.1"),
	  version(1.0),
	  uuid(guid),
	  helpstring("nprof COR Profiling Hook Class")
	]
	class ATL_NO_VTABLE CNProfCORHook : public INProfCORHook,public ICorProfilerCallback2 {
	public:
		CNProfCORHook() 
		{
			this->profiler = 0;
		}
		DECLARE_PROTECT_FINAL_CONSTRUCT()
		HRESULT FinalConstruct() 
		{
			return S_OK;
		}
		void FinalRelease() 
		{
			delete profiler;
		}
	public:
		static Profiler* profiler;
		CRITICAL_SECTION criticalSection;
	public:
		static Profiler* GetProfiler() 
		{
			return profiler;
		}
		STDMETHOD(Initialize)(LPUNKNOWN pICorProfilerInfoUnk) 
		{
			//cout << "Hello from hook" << endl;
			//Sleep(15000);
			//cout << "Go!" << endl;
			CComQIPtr<ICorProfilerInfo2> profilerInfo = pICorProfilerInfoUnk;
			InitializeCriticalSection(&criticalSection);
			profiler = new Profiler(profilerInfo);
			return S_OK;
		}
		STDMETHOD(Shutdown)() 
		{
			//int a;
			//cin >> a;
			EnterCriticalSection(&criticalSection);
			profiler->End();
			delete profiler;
			profiler = 0;
			LeaveCriticalSection(&criticalSection);
			return S_OK;
		}
		STDMETHOD(AppDomainCreationStarted)(AppDomainID appDomainId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AppDomainCreationFinished)(AppDomainID appDomainId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AppDomainShutdownStarted)(AppDomainID appDomainId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AppDomainShutdownFinished)(AppDomainID appDomainId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AssemblyLoadStarted)(AssemblyID assemblyId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AssemblyLoadFinished)(AssemblyID assemblyId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AssemblyUnloadStarted)(AssemblyID assemblyId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(AssemblyUnloadFinished)(AssemblyID assemblyId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ModuleLoadStarted)(ModuleID moduleId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ModuleLoadFinished)(ModuleID moduleId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ModuleUnloadStarted)(ModuleID moduleId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ModuleUnloadFinished)(ModuleID moduleId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ModuleAttachedToAssembly)(ModuleID moduleId, AssemblyID assemblyId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ClassLoadStarted)(ClassID classId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ClassLoadFinished)(ClassID classId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ClassUnloadStarted)(ClassID classId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ClassUnloadFinished)(ClassID classId, HRESULT hrStatus) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(FunctionUnloadStarted)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITCompilationStarted)(FunctionID functionId, BOOL fIsSafeToBlock) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITCompilationFinished)(FunctionID functionId, HRESULT hrStatus, BOOL fIsSafeToBlock) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITCachedFunctionSearchStarted)(FunctionID functionId, BOOL* pbUseCachedFunction) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITCachedFunctionSearchFinished)(FunctionID functionId, COR_PRF_JIT_CACHE result) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITFunctionPitched)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(JITInlining)(FunctionID callerId, FunctionID calleeId, BOOL* pfShouldInline) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ThreadCreated)(ThreadID threadId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ThreadDestroyed)(ThreadID threadId) 
		{
			EnterCriticalSection(&criticalSection);
			profiler->ThreadDestroyed( threadId );
			LeaveCriticalSection(&criticalSection);
			return S_OK;
		}
		STDMETHOD(ThreadAssignedToOSThread)(ThreadID managedThreadId, DWORD osThreadId) 
		{
			EnterCriticalSection(&criticalSection);
			profiler->ThreadAssigned( managedThreadId, osThreadId );
			LeaveCriticalSection(&criticalSection);
			return S_OK;
		}
		STDMETHOD(RemotingClientInvocationStarted)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingClientSendingMessage)(GUID * pCookie, BOOL fIsAsync) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingClientReceivingReply)(GUID * pCookie, BOOL fIsAsync) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingClientInvocationFinished)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingServerReceivingMessage)(GUID * pCookie, BOOL fIsAsync) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingServerInvocationStarted)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingServerInvocationReturned)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RemotingServerSendingReply)(GUID * pCookie, BOOL fIsAsync) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(UnmanagedToManagedTransition)(FunctionID functionId, COR_PRF_TRANSITION_REASON reason) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ManagedToUnmanagedTransition)(FunctionID functionId, COR_PRF_TRANSITION_REASON reason) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeSuspendStarted)(COR_PRF_SUSPEND_REASON suspendReason) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeSuspendFinished)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeSuspendAborted)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeResumeStarted)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeResumeFinished)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeThreadSuspended)(ThreadID threadId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RuntimeThreadResumed)(ThreadID threadId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(MovedReferences)(unsigned long cMovedObjectIDRanges, ObjectID* oldObjectIDRangeStart, ObjectID* newObjectIDRangeStart, unsigned long * cObjectIDRangeLength) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ObjectAllocated)(ObjectID objectId, ClassID classId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ObjectsAllocatedByClass)(unsigned long cClassCount, ClassID* classIds, unsigned long* cObjects) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ObjectReferences)(ObjectID objectId, ClassID classId, unsigned long cObjectRefs, ObjectID* objectRefIds) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RootReferences)(unsigned long cRootRefs, ObjectID* rootRefIds) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionThrown)(ThreadID thrownObjectId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionSearchFunctionEnter)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionSearchFunctionLeave)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionSearchFilterEnter)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionSearchFilterLeave)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionSearchCatcherFound)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionOSHandlerEnter)(UINT_PTR __unused) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionOSHandlerLeave)(UINT_PTR __unused) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionUnwindFunctionEnter)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionUnwindFunctionLeave)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionUnwindFinallyEnter)(FunctionID functionId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionUnwindFinallyLeave)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionCatcherEnter)(FunctionID functionId, ObjectID objectId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionCatcherLeave)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(COMClassicVTableCreated)(ClassID wrappedClassId, const GUID& implementedIID, void * pVTable, unsigned long cSlots) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(COMClassicVTableDestroyed)(ClassID wrappedClassId, const GUID& implementedIID, void * pVTable) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionCLRCatcherFound)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ExceptionCLRCatcherExecute)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(ThreadNameChanged)(ThreadID threadId, ULONG cchName, WCHAR name[]) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(GarbageCollectionStarted)(int cGenerations, BOOL generationCollected[], COR_PRF_GC_REASON reason) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(SurvivingReferences) (ULONG cSurvivingObjectIDRanges,ObjectID objectIDRangeStart[],ULONG cObjectIDRangeLength[]) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(GarbageCollectionFinished)() 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(FinalizeableObjectQueued)(DWORD finalizerFlags,ObjectID objectID) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(RootReferences2)(ULONG cRootRefs, ObjectID rootRefIds[], COR_PRF_GC_ROOT_KIND rootKinds[],COR_PRF_GC_ROOT_FLAGS rootFlags[], UINT_PTR rootIds[]) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(HandleCreated)(GCHandleID handleId,ObjectID initialObjectId) 
		{
			return E_NOTIMPL;
		}
		STDMETHOD(HandleDestroyed)(GCHandleID handleId) 
		{
			return E_NOTIMPL;
		}
	};
	Profiler* CNProfCORHook::profiler;
}
[ 
	module(dll, uuid = "{A461E20A-C7DC-4A89-A24E-87B5E975A96B}", 
	name = "NProfHook", 
	helpstring = "NProf.Hook 1.0 Type Library",	
	resource_name = "IDR_NPROFHOOK")
];