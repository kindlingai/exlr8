/***************************************************************************
                          profiler.cpp  -  description
                             -------------------
    begin                : Sat Jan 18 2003
    copyright            : (C) 2003,2004,2005,2006 by Matthew Mastracci, Christian Staudenmeyer
    email                : mmastrac@canada.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Serialization;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using System.Reflection;
using System.Xml;

namespace NProf
{
    public class Test
    {
        public static void Main()
        {
			Dictionary<string,string> dictionary=new Dictionary<string,string>();
			//list.AddRange(new string[] { "", "", "", "", "", "", "", "", "", "" });
            for (int i = 0; i < 50000000; i++)
            {
                object x = new object();
				dictionary["hello"] = "world";
				//XmlDocument a=new XmlDocument();
				//list[i % 10] = "hello";
            }            
        }
    }
}
