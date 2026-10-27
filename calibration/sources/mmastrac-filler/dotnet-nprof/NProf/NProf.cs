/***************************************************************************
                         profiler.cpp  -  description
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

using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;
using System.Data;
using Microsoft.Win32;
using System.Globalization;
using System.Reflection;

namespace NProf
{
    using FunctionID = System.UInt64;
    public class UpDownButton : Button
    {
        SortColumn sort;
        public UpDownButton(string text, SortColumn sort, RunView runView)
        {
            this.Text = text;
            this.sort = sort;

            this.Font = new Font("Microsoft Sans Serif", 8);
            this.Padding = new Padding(0);
            this.Margin = new Padding(0);
            this.Click += delegate
            {
                runView.calls.Nodes.Clear();
                runView.sort = sort;
                runView.Update(runView.currentRun);
            };
        }
    }
    public class SearchPanel : FlowLayoutPanel
    {
        public void Toggle()
        {
            Visible = !Visible;
            if (Visible)
            {
                this.findText.Focus();
            }
            else
            {
                this.runView.calls.Focus();
            }
        }
        public void Find(bool forward, bool step)
        {
            if (methodView.SelectedNode != null)
            {
                runView.Find(findText.Text, forward, step);
            }
            else
            {
                runView.Find(findText.Text, forward, step);
            }
        }
        public View methodView;
        public RunView runView;
        public SearchPanel(View methodView, RunView runView)
        {
            this.runView = runView;
            Dock = DockStyle.Bottom;
            this.methodView = methodView;
            findText = new TextBox();
            findText.AcceptsReturn = true;
            findText.TextChanged += delegate
            {
                Find(true, false);
            };
            findText.KeyDown += delegate(object sender, KeyEventArgs e)
            {
                if (e.KeyCode == Keys.Enter)
                {
                    Find(true, true);
                    e.Handled = true;
                    e.SuppressKeyPress = true;
                }
                else if (e.KeyCode == Keys.Escape)
                {
                    Toggle();
                }
                if (e.KeyData == (Keys.F | Keys.Control))
                {
                    Toggle();
                }
            };
            BorderStyle = BorderStyle.FixedSingle;
            WrapContents = false;
            AutoSize = true;
            Button findNext = new Button();
            Button close = new Button();
            close.Image = new Bitmap(this.GetType().Assembly.GetManifestResourceStream("NProf.Resources.close.gif"));
            close.Click += delegate { Toggle(); };
            close.AutoSize = true;
            close.Width = 30;

            findNext.AutoSize = true;
            findNext.Text = "&Next";
            findNext.Click += delegate(object sender, EventArgs e)
            {
                Find(true, true);
            };

            Label findLabel = new Label();
            findLabel.Text = "Find:";
            findLabel.Dock = DockStyle.Fill;
            findLabel.TextAlign = ContentAlignment.MiddleLeft;
            findLabel.AutoSize = true;

            Button findPrevious = new Button();
            findPrevious.AutoSize = true;
            findPrevious.FlatAppearance.BorderSize = 0;

            findPrevious.Click += delegate(object sender, EventArgs e)
            {
                Find(false, true);
            };
            findPrevious.Text = "&Previous";
            Controls.AddRange(new Control[] { findLabel, findText, findNext, findPrevious, close });
            this.Visible = false;
        }

        public TextBox findText;
    }
    public enum SortColumn
    {
        Inclusive,
        Exclusive,
        InclusiveChange,
        ExclusiveChange
    }
    public class RunView : TabPage
    {
        //public bool isEmpty;
        Button toggle = new Button();
        //public TextBox application;
        public SearchPanel searchPanel;
        public void CheckFind(KeyEventArgs e)
        {
            if (e.KeyData == (Keys.F | Keys.Control))
            {
                searchPanel.Toggle();
                e.Handled = true;
                e.SuppressKeyPress = true;
                if (!searchPanel.Visible)
                {
                    calls.Focus();
                }
            }
        }
        public View calls = new View();
        public TreeNode MoveTo(FunctionID id)
        {
            calls.SelectedNode = calls.Nodes[id.ToString()];
            return calls.SelectedNode;
        }
        void MethodView_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                calls.SelectedNode = calls.GetNodeAt(e.X, e.Y);
            }
        }
        public void Update(Run run)
        {
            currentRun = run;
            calls.SuspendLayout();
            Invalidate();
            calls.BeginUpdate();
            calls.Nodes.Clear();
            FunctionMap f = showCallees ? run.functions : run.callers;
            foreach (Function method in SortFunctions(f.Values))
            {
                AddItem(calls.Nodes, method);
            }

            //foreach (Function method in SortFunctions(functions.Values))
            //{
            //    AddItem(calls.Nodes, method);
            //}
            foreach (TreeNode item in calls.Nodes)
            {
                EnsureComputed(item);
            }
            calls.EndUpdate();
            calls.ResumeLayout();
            calls.Focus();
        }
        public void Find(string text, bool forward, bool step)
        {
            if (text != "")
            {
                TreeNode item;
                if (calls.SelectedNode == null)
                {
                    if (calls.Nodes.Count == 0)
                    {
                        item = null;
                    }
                    else
                    {
                        item = calls.Nodes[0];
                    }
                }
                else
                {
                    if (step)
                    {
                        if (forward)
                        {
                            item = calls.SelectedNode.NextVisibleNode;
                        }
                        else
                        {
                            item = calls.SelectedNode.PrevVisibleNode;
                        }
                    }
                    else
                    {
                        item = calls.SelectedNode;
                    }
                }
                TreeNode firstItem = item;
                while (item != null)
                {
                    if (item.Text.ToLower().Contains(text.ToLower()))
                    {
                        calls.SelectedNode = null;
                        calls.SelectedNode = item;
                        this.Invalidate();
                        break;
                    }
                    else
                    {
                        if (forward)
                        {
                            item = item.NextVisibleNode;
                        }
                        else
                        {
                            item = item.PrevVisibleNode;
                        }
                        if (item == null)
                        {
                            if (forward)
                            {
                                item = calls.Nodes[0];
                            }
                            else
                            {
                                item = calls.Nodes[calls.Nodes.Count - 1];
                            }
                        }
                    }
                    if (item == firstItem)
                    {
                        break;
                    }
                }
                if (item != null)
                {
                }
            }
        }
        public Run currentRun;
        void EnsureComputed(TreeNode item)
        {
            MakeSureComputed(item, true);
        }
        public List<Function> SortFunctions(IEnumerable<Function> f)
        {
            List<Function> functions = new List<Function>(f);
            functions.Sort(delegate(Function a, Function b)
            {
                switch (sort)
                {
                    case SortColumn.Inclusive:
                        return b.Samples.CompareTo(a.Samples);
                    case SortColumn.Exclusive:
                        return b.ExclusiveSamples.CompareTo(a.ExclusiveSamples);
                    case SortColumn.InclusiveChange:
                        return a.InclusiveChange.CompareTo(b.InclusiveChange);
                    case SortColumn.ExclusiveChange:
                        return a.ExclusiveChange.CompareTo(b.ExclusiveChange);
                }
                return b.Samples.CompareTo(a.Samples);
            });
            return functions;
        }
        void MakeSureComputed(TreeNode item, bool parentExpanded)
        {
            if (item.Nodes.Count == 0 && item.Tag != null)
            {
                Function f = ((Function)item.Tag);
                foreach (Function function in SortFunctions(f.Children.Values))
                {
                    AddFunctionItem(item.Nodes, function);
                }
                //if (f.ExclusiveSamples != 0 && item.Nodes.Count!=0)
                //{
                //    item.Nodes.Add(FormatNumber(100*(double)f.ExclusiveSamples/(double)run.maxSamples).PadLeft(4,' ') + " method body");
                //}
            }
            if (item.IsExpanded || parentExpanded)
            {
                foreach (TreeNode subItem in item.Nodes)
                {
                    MakeSureComputed(subItem, item.IsExpanded);
                }
            }
        }
        private Function GetOtherFunction(Function function, FunctionMap other)
        {
            List<Function> functions = new List<Function>();
            Function f = function;
            while (f != null)
            {
                functions.Add(f);
                f = f.parent;
                if (functions.Count > 10)
                {
                }
                if (functions.Count > 100)
                {
                    break;
                }
                //break;
            }
            functions.Reverse();
            if (!other.ContainsKey(functions[0].ID))
            {
                return null;
            }

            Function start = other[functions[0].ID];// ((Function)other.Nodes[functions[0].ToString()].Tag);
            functions.RemoveAt(0);

            Function o = start;
            foreach (Function a in functions)
            {
                if (o.Children.ContainsKey(a.ID))
                {
                    o = o.Children[a.ID];
                }
                else
                {
                    return null;
                }
            }
            return o;
            //if (function.parent == null)
            //{
            //    if (other.Children.ContainsKey(function.ID))
            //    {
            //        return other.Children[function.ID];
            //    }
            //    else
            //    {
            //        return null;
            //    }
            //}
            //else
            //{
            //    Function f=GetOtherFunction(function.parent, other);
            //    if (f != null)
            //    {
            //        if(other.Children.ContainsKey(
            //    }
            //    else
            //    {
            //        return null;
            //    }
            //}
        }
        private TreeNode AddItem(TreeNodeCollection parent, Function function)
        {
            string signature = currentRun.signatures[function.ID].Signature;
            if (parent[signature] != null)
            {
                return parent[signature];
            }
            else
            {
                double fraction = ((double)function.Samples) / (double)currentRun.maxSamples;
                double percent = (100.0 * ((double)function.Samples / (double)function.run.maxSamples));
                double exclusivePercent = (100.0 * ((double)function.ExclusiveSamples / (double)function.run.maxSamples));
                string text = FormatNumber(percent).PadLeft(3, ' ');
                if (comparisonRun != null)
                {

                    if (comparisonRun.functions.ContainsKey(function.ID))
                    {
                        //while (true)
                        //{
                        Function other = GetOtherFunction(function, comparisonRun.functions);//[function.ID]);
                        //Function other = GetOtherFunction(function, comparisonRun.functions[0]);
                        //}
                        //Function other = comparisonRun.functions[function.ID];
                        if (other == null)
                        {
                            function.InclusiveChange = (double)function.Samples / (double)function.run.maxSamples;
                        }
                        else
                        {
                            function.InclusiveChange = (((double)function.Samples - (double)other.Samples) / function.run.maxSamples) * 100;
                        }
                        //function.ExclusiveChange = (((double)function.ExclusiveSamples - (double)other.ExclusiveSamples) / function.run.maxSamples) * 100;
                    }
                    else
                    {
                        function.InclusiveChange = fraction;
                        //function.ExclusiveChange = exclusivePercent;
                    }
                    text += " " + (function.InclusiveChange).ToString("+0;-0;+0").PadLeft(3, ' ');
                }
                text += " " + currentRun.signatures[function.ID].Signature.Trim();
                TreeNode item = new TreeNode();
                item.Tag = function;
                item.Name = function.ID.ToString();
                item.Text = text;
                parent.Add(item);
                return item;
            }
        }

        private static string FormatNumber(double percent)
        {
            //if (percent == 0)
            //{
            //    return "";
            //}
            return (percent).ToString("0;-0;0");
        }
        //private static string FormatChange(double change)
        //{
        //    return ((change < 0 ? "-" : "+") + FormatNumber(change)).PadLeft(6, ' ');
        //}
        public void AddFunctionItem(TreeNodeCollection parent, Function function)
        {
            TreeNode item = AddItem(parent, function);
            foreach (Function callee in SortFunctions(function.Children.Values))
            {
                AddItem(item.Nodes, callee);
            }
        }
        public void Add(Function function)
        {
            AddFunctionItem(calls.Nodes, function);
        }
        public ComboBox comparison = new ComboBox();
        public static Label Caption(string text)
        {
            Label caption = new Label();
            caption.Text = text;
            caption.Dock = DockStyle.Top;
            caption.TextAlign = ContentAlignment.MiddleLeft;
            return caption;
        }
        public static int count = 0;
        public void UpdateComparison()
        {
            comparison.Items.Clear();
            comparison.Items.Add("");

            foreach (RunView r in NProf.tabs.TabPages)
            {
                if (r != this)
                {
                    comparison.Items.Add(r);
                }
            }
        }
        public bool showCallees = true;
        public RunView(Run run)//:this(run,false)
        {
            count++;
            this.Text = Path.GetFileNameWithoutExtension(run.Executable) + " #" + count;
            calls.MouseClick += new MouseEventHandler(MethodView_MouseClick);
            calls.Dock = DockStyle.Fill;
            calls.Dock = DockStyle.Top;
            calls.BeforeExpand += delegate(object sender, TreeViewCancelEventArgs e)
            {
                EnsureComputed(e.Node);
            };
            calls.HideSelection = false;
            this.run = run;
            ContextMenu callsMenu = new ContextMenu(new MenuItem[] {
				new MenuItem("Go to method",delegate(object sender,EventArgs e){
					MoveTo(((Function)calls.SelectedNode.Tag).ID);
				}),
				new MenuItem("Show callers",delegate(object sender,EventArgs e){
					FunctionID id=((Function)calls.SelectedNode.Tag).ID;
					ToggleCalls();
					MoveTo(id);
				})
			});
            ContextMenu callersMenu = new ContextMenu(new MenuItem[] {
				new MenuItem("Go to method",delegate(object sender,EventArgs e){
					MoveTo(((Function)calls.SelectedNode.Tag).ID);
				}),
				new MenuItem("Show callees",delegate(object sender,EventArgs e){
					FunctionID id=((Function)calls.SelectedNode.Tag).ID;
					ToggleCalls();
					MoveTo(id);
				})
			});
            calls.MouseClick += delegate(object sender, MouseEventArgs e)
            {
                if (e.Button == MouseButtons.Right)
                {
                    if (showCallees)
                    {
                        callsMenu.Show(calls, new Point(e.X, e.Y));
                    }
                    else
                    {
                        callersMenu.Show(calls, new Point(e.X, e.Y));
                    }
                }
            };
            calls.Size = new Size(100, 100);
            calls.Dock = DockStyle.Fill;

            comparison.DropDownStyle = ComboBoxStyle.DropDownList;
            comparison.Dock = DockStyle.Top;

            this.UpdateComparison();


            comparison.DisplayMember = "Text";
            comparison.Width = 100;

            if (comparison.Items.Count != 0)
            {
                comparison.SelectedIndexChanged += new EventHandler(comparison_SelectedIndexChanged);
                comparison.SelectedIndex = 0;

            }
            if (run != null)
            {
                Update(run);
                //Update(run, run.callers);
            }
            Controls.Add(calls);

            searchPanel = new SearchPanel(calls, this);
            calls.KeyDown += delegate(object sender, KeyEventArgs e)
            {
                CheckFind(e);
            };

            searchPanel.Dock = DockStyle.Bottom;
            FlowLayoutPanel q = new FlowLayoutPanel();
            q.Padding = new Padding(0);
            //q.BackColor = Color.Red;
            q.Margin = new Padding(0);
            q.Dock = DockStyle.Top;
            q.Height = 25;
            Label inclusive = MakeLabel("Time");

            Graphics graphics = this.CreateGraphics();
            graphics.PageUnit = GraphicsUnit.Pixel;
            SizeF size = graphics.MeasureString("a", NProf.font, 100, StringFormat.GenericTypographic);

            //Label label = new Label();
            //label.Width = (int)size.Width * 3;



            //q.Controls.Add(label);



            toggle.Text = "Callees";
            toggle.Click += delegate
            {
                ToggleCalls();

            };
            q.Controls.Add(toggle);

            int s = (int)(size.Width * 7);
            inclusive.Width = s;
            //q.Controls.Add(inclusive);
            Label inclusiveChanged = MakeLabel("Change");
            inclusiveChanged.Width = s;
            inclusiveChanged.Visible = false;
            q.Controls.Add(inclusiveChanged);
            //q.Controls.Add(MakeLabel("Signature"));


            comparison.SelectedIndexChanged += delegate
            {
                inclusiveChanged.Visible = comparison.SelectedIndex != 0;
            };
            //Controls.Add(q);
            //FlowLayoutPanel p = new FlowLayoutPanel();
            //p.Dock = DockStyle.Top;
            //p.Height = 30;
            Controls.Add(searchPanel);
            //Controls.Add(p);
            Label l = new Label();
            l.Width = (int)size.Width;
            //p.Controls.Add(l);

            this.Dock = DockStyle.Fill;

            //application = new TextBox();
            //application.TextChanged += delegate
            //{
            //    string fileName = Path.GetFileName(application.Text);
            //    Text = fileName + " - ";// +Title;
            //    //Text = fileName + " - " + Title;
            //};

            //application.Width = 200;
            //application.Dock = DockStyle.Fill;
            //start = new Button();
            //start.Text = "";
            //start.Click += delegate
            //{
            //    StartRun();
            //};
            //application.Text = Settings.Application;
            //start.Text = "Start";
            //start.Anchor = AnchorStyles.Right;
            //start.BackgroundImageLayout = ImageLayout.Zoom;
            //start.Width = start.PreferredSize.Width;

            //Button browse = new Button();
            //browse.Anchor = AnchorStyles.Top;
            //browse.Text = "...";
            //browse.Width = 0;
            //browse.AutoSize = true;
            //browse.Focus();
            //browse.TextAlign = ContentAlignment.MiddleCenter;
            //browse.Click += delegate
            //{
            //    OpenFileDialog dialog = new OpenFileDialog();
            //    dialog.FileName = application.Text;
            //    dialog.Filter = "Executable files (*.exe)|*.exe";
            //    DialogResult dr = dialog.ShowDialog();
            //    if (dr == DialogResult.OK)
            //    {
            //        application.Text = dialog.FileName;
            //        application.Focus();
            //        application.SelectAll();
            //    }
            //};
            //p.Controls.Add(MakeLabel("Application:"));
            //p.Controls.Add(application);
            //p.Controls.Add(browse);
            //p.Controls.Add(start);
            //p.Controls.Add(MakeLabel("Compare to:"));
            //p.Controls.Add(comparison);

        }
        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.KeyCode == (Keys.Tab | Keys.Control))
            {
                e.Handled = true;
            }
            base.OnKeyDown(e);
        }
        //public RunView(Run run,bool isEmpty)
        //{			
        //    this.isEmpty = isEmpty;
        //    if (isEmpty)
        //    {
        //        NProf.tabs.Selected += new TabControlEventHandler(tabs_Selected);
        //    }
        //    else
        //    {
        //        Initialize(run);
        //    }



        //}

        void tabs_Selected(object sender, TabControlEventArgs e)
        {
            //if (e.TabPage == this)
            //{
            //    if (isEmpty)
            //    {
            //        isEmpty = false;
            //        NProf.tabs.TabPages.Add(new RunView(null, true));
            //        Initialize(null);
            //    }
            //}
        }



        private void ToggleCalls()
        {
            showCallees = !showCallees;
            if (showCallees)
            {
                toggle.Text = "Callees";
            }
            else
            {
                toggle.Text = "Callers";
            }
            Update(this.run);
        }
        private static Label MakeLabel(string text)
        {
            Label label = new Label();
            label.Text = text;
            label.Dock = DockStyle.Fill;
            label.TextAlign = ContentAlignment.MiddleLeft;
            label.AutoSize = true;
            return label;
        }

        // TODO: remove
        public SortColumn sort = SortColumn.Inclusive;

        public Run run;

        void comparison_SelectedIndexChanged(object sender, EventArgs e)
        {
            RunView r = comparison.Items[comparison.SelectedIndex] as RunView;
            if (r == null)
            {
                comparisonRun = null;
            }
            else
            {
                comparisonRun = r.run;
            }
            calls.Nodes.Clear();
            if (run != null)
            {
                Update(run);
            }
        }
        public Run comparisonRun;
    }
    public class CustomTabControl : TabControl
    {
        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.KeyData == (Keys.Tab | Keys.Control))
            {
                if (this.SelectedIndex < this.TabPages.Count - 2)
                {
                    this.SelectedIndex++;
                }
                else
                {
                    this.SelectedIndex = 0;
                }
                e.Handled = true;
            }
            else if (e.KeyData == (Keys.Tab | Keys.Control | Keys.Shift))
            {
                if (this.SelectedIndex > 0)
                {
                    this.SelectedIndex--;
                }
                else
                {
                    this.SelectedIndex = this.TabPages.Count - 2;
                }
                e.Handled = true;
            }
            else
            {
                base.OnKeyDown(e);
            }
        }
    }
    public class NProf : Form
    {
        public void StartRun()
        {
            //int length = Path.ChangeExtension(application.Text, "abc").Length;
            string fileName = application.Text;//.Substring(0, length);
            //string arguments = argum.Text.Substring(length);
            if (!File.Exists(fileName))
            {
                MessageBox.Show("Please select an executable to profile.", "NProf", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            Run run = new Run(NProf.form.profiler, fileName);
            run.profiler.completed = new EventHandler(run.Complete);
            run.Start(fileName, arguments.Text);
        }
        private static Label MakeLabel(string text)
        {
            Label label = new Label();
            label.Text = text;
            label.Dock = DockStyle.Fill;
            label.TextAlign = ContentAlignment.MiddleLeft;
            label.AutoSize = true;
            return label;
        }
        public static Font font = new Font("Courier New", 12.0f);
        public Profiler profiler;
        public static TextBox application;
        public static TextBox arguments;
        public static TextBox directory;
        public Button start;

        public string Title
        {
            get
            {
                return "NProf " + Version;
            }
        }
        FlowLayoutPanel mainPanel = new FlowLayoutPanel();
        Label help = MakeLabel("Select the application to profile and click 'Start'.");
        private NProf()
        {
            this.Closing += delegate
            {
                Settings.Application = application.Text;
                Settings.Arguments = arguments.Text;
                Settings.WorkingDirectory = directory.Text;
                if (tabs.SelectedTab != null)
                {
                    Settings.Application = application.Text;
                }
                SaveSettings();
            };
            LoadSettings();
            help.Font = new Font("Microsoft Sans Serif", 10);
            WindowState = FormWindowState.Maximized;
            Icon = new Icon(this.GetType().Assembly.GetManifestResourceStream("NProf.Resources.app-icon.ico"));
            Text = Title;
            profiler = new Profiler();
            Menu = new MainMenu(new MenuItem[] {
				new MenuItem(
					"File",
					new MenuItem[] {
                        //new MenuItem(
                        //    "New Tab",
                        //    delegate {
                        //        //RunView v=new RunView(null);
                        //        //tabs.TabPages.Add(v);
                        //        //tabs.SelectedTab=v;
                        //        tabs.SelectedIndex=tabs.TabPages.Count-1;
                        //        //v.Focus();
                        //    },
                        //    Shortcut.CtrlT
                        //),
						new MenuItem(
							"Close Tab",
							delegate {RemoveTab();},Shortcut.CtrlF4
						),
						new MenuItem(
							"E&xit",
							delegate {Close();}
						)
					}
				),
				new MenuItem(
				    "Edit",
				    new MenuItem[] {
			            new MenuItem("Find",delegate{
			                if(tabs.SelectedTab!=null)
			                {
								((RunView)tabs.SelectedTab).searchPanel.Toggle();
			                }
			            },
						Shortcut.CtrlF
						),
			            new MenuItem(
							"Start Profiling",
							delegate{
								//if(tabs.SelectedTab!=null)
								//{
									StartRun();
									//((RunView)tabs.SelectedTab).StartRun();
								//}
							},
							Shortcut.F5
						)
				    }
				)
            });
            tabs = new CustomTabControl();
            mainPanel.AutoSize = true;
            application = new TextBox();
            application.Width = 200;
            application.Dock = DockStyle.Fill;
            arguments = new TextBox();
            arguments.Width = 200;
            directory = new TextBox();

            start = new Button();
            start.Text = "";
            start.Click += delegate
            {
                StartRun();
            };
            application.Text = Settings.Application;
            arguments.Text = Settings.Arguments;
            directory.Text = Settings.WorkingDirectory;
            start.Text = "Start";
            start.Anchor = AnchorStyles.Right;
            start.BackgroundImageLayout = ImageLayout.Zoom;
            directory.Width = 200;
            start.Width = start.PreferredSize.Width;
            mainPanel.Height = 100;
            mainPanel.AutoSize = true;
            mainPanel.Dock = DockStyle.Top;
            start.TextImageRelation = TextImageRelation.TextAboveImage;

            Button browse = new Button();
            browse.Anchor = AnchorStyles.Top;
            browse.Text = "Browse...";
            browse.Focus();
            browse.TextAlign = ContentAlignment.MiddleCenter;
            browse.Click += delegate
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.FileName = application.Text;
                dialog.Filter = "Executable files (*.exe)|*.exe";
                DialogResult dr = dialog.ShowDialog();
                if (dr == DialogResult.OK)
                {
                    application.Text = dialog.FileName;
                    application.Focus();
                    application.SelectAll();
                }
            };
            Button directoryBrowse = new Button();
            directoryBrowse.Text = "Browse...";
            directoryBrowse.Click += delegate
            {
                FolderBrowserDialog dialog = new FolderBrowserDialog();
                dialog.SelectedPath = directory.Text;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    directory.Text = dialog.SelectedPath;
                }
            };
            mainPanel.Controls.Add(MakeLabel("Application:"));
            mainPanel.Controls.Add(application);
            mainPanel.Controls.Add(browse);
            mainPanel.Controls.Add(MakeLabel("Arguments:"));
            mainPanel.Controls.Add(arguments);
            mainPanel.Controls.Add(MakeLabel("Working directory:"));
            mainPanel.Controls.Add(directory);
            mainPanel.Controls.Add(directoryBrowse);
            mainPanel.Controls.Add(start);
            help.Dock = DockStyle.Fill;
            Controls.Add(help);
            Controls.Add(mainPanel);

            mainPanel.Padding = new Padding(3);
            tabs.SelectedIndexChanged += new EventHandler(tabs_SelectedIndexChanged);
            tabs.Dock = DockStyle.Fill;
            help.Dock = DockStyle.Fill;
            help.TextAlign = ContentAlignment.MiddleCenter;
            help.AutoSize = false;

            tabs.KeyDown += new KeyEventHandler(tabs_KeyDown);

            InitializeComponent();
            //tabs.DrawMode = System.Windows.Forms.TabDrawMode.OwnerDrawFixed;
            //tabs.DrawItem += TabControl1_DrawItem;
            //tabs.MouseClick += TabControl1_MouseClick;
            //tabs.MouseHover += new EventHandler(tabs_MouseHover);
            img = new Bitmap(GetType().Assembly.GetManifestResourceStream("NProf.Resources.close.gif"));
            //tabs.TabPages.Add(new RunView(null));
            //tabs.TabPages.Add(new RunView(null,true));
            //Controls.Add(tabs);
        }

        void tabs_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyData == (Keys.Tab | Keys.Control))
            {
                e.Handled = true;
            }
        }
        void tabs_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (tabs.SelectedTab != null)
            {
                ((RunView)tabs.SelectedTab).UpdateComparison();
            }
        }
        void tabs_MouseHover(object sender, EventArgs e)
        {
            Rectangle r = tabs.GetTabRect(tabs.SelectedIndex);
            ToolTip t = new ToolTip();
            Point p = PointToClient(Cursor.Position) - new Size(tabs.Left, tabs.Top);
            if (TestHit(tabs, p))
            {
                t.Show("Close tab", tabs, r.Right, 0, 1000);
            }
        }
        private Point _imageLocation = new Point(15, 7);
        private Point _imgHitArea = new Point(13, 2);
        Image img;
        private void TabControl1_DrawItem(object sender, System.Windows.Forms.DrawItemEventArgs e)
        {
            try
            {
                Rectangle r = e.Bounds;
                r = tabs.GetTabRect(e.Index);
                r.Offset(2, 2);
                Brush TitleBrush = new SolidBrush(Color.Black);
                Font f = this.Font;
                string title = tabs.TabPages[e.Index].Text;
                e.Graphics.DrawString(title, f, TitleBrush, new PointF(r.X, r.Y));
                //if (!((RunView)tabs.TabPages[e.Index]).isEmpty)
                //{
                e.Graphics.DrawImage(img, new Point(r.X + (tabs.GetTabRect(e.Index).Width - _imageLocation.X), _imageLocation.Y));
                //}
            }
            catch (Exception)
            {
            }
        }
        private void TabControl1_MouseClick(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            TabControl tc = (TabControl)sender;
            Point p = e.Location;
            if (TestHit(tc, p))
            {
                RemoveTab();
            }
        }
        public void RemoveTab()
        {
            int i = tabs.SelectedIndex;
            if (tabs.SelectedIndex < tabs.TabPages.Count - 2)
            {
                tabs.SelectedIndex++;
            }
            else if (tabs.SelectedIndex > 0)
            {
                tabs.SelectedIndex--;
            }
            tabs.TabPages.RemoveAt(i);
        }
        private bool TestHit(TabControl tc, Point p)
        {
            int _tabWidth = 0;
            _tabWidth = tabs.GetTabRect(tc.SelectedIndex).Width - (_imgHitArea.X);
            Rectangle r = tabs.GetTabRect(tc.SelectedIndex);
            r.Offset(_tabWidth, _imgHitArea.Y);
            r.Width = 16;
            r.Height = 16;
            if (r.Contains(p))
            {
                return true;
            }
            return false;
        }

        public static TabControl tabs;
        public void ShowRun(Run run)
        {
            RunView runView = new RunView(run);
            if (!Controls.Contains(tabs))
            {
                Controls.Remove(mainPanel);
                Controls.Remove(help);
                Controls.Add(tabs);
                Controls.Add(mainPanel);
            }
            tabs.Controls.Add(runView);
            tabs.SelectedTab = runView;
        }
        [STAThread]
        static void Main(string[] args)
        {
            string s = Guid.NewGuid().ToString().ToUpper();
            Application.EnableVisualStyles();
            Application.Run(form);
        }
        public static NProf form = new NProf();
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(NProf));
            this.SuspendLayout();
            // 
            // NProf
            // 
            this.ClientSize = new System.Drawing.Size(292, 273);
            this.Name = "NProf";
            this.ResumeLayout(false);

        }
        public static string ProgramName
        {
            get
            {
                return "NProf";
            }
        }
        public static string Version
        {
            get
            {
                return "0.12";
            }
        }
        public static void LoadSettings()
        {
            using (RegistryKey key = Registry.CurrentUser.OpenSubKey("Software", true).CreateSubKey(ProgramName).CreateSubKey(Version))
            {
                foreach (FieldInfo field in typeof(Settings).GetFields())
                {
                    if (field.FieldType == typeof(string[]))
                    {
                        List<string> values = new List<string>();
                        foreach (string k in key.OpenSubKey(field.Name).GetValueNames())
                        {
                            values.Add((string)key.OpenSubKey(field.Name).GetValue(k));
                        }
                        field.SetValue(null, values.ToArray());
                    }
                    object value = key.GetValue(field.Name, null);
                    if (value != null)
                    {
                        field.SetValue(null, value);
                    }
                }
            }
        }
        public static void SaveSettings()
        {
            using (RegistryKey key = Registry.CurrentUser.OpenSubKey("Software", true).CreateSubKey(ProgramName).CreateSubKey(Version))
            {
                foreach (FieldInfo field in typeof(Settings).GetFields())
                {
                    object value = field.GetValue(null);
                    if (value is string[])
                    {
                        key.DeleteSubKeyTree(field.Name);
                        RegistryKey k = key.CreateSubKey(field.Name);
                        int count = 1;
                        foreach (string s in (string[])value)
                        {
                            k.SetValue(count.ToString(), s);
                            count++;
                        }
                    }
                    else
                    {
                        if (value != null)
                        {
                            key.SetValue(field.Name, value);
                        }
                    }
                }
            }
        }
    }
    public class Settings
    {
        public static string Application = "";
        public static string Arguments = "";
        public static string WorkingDirectory = "";
    }
    public class StackWalk
    {
        public StackWalk(int id, int index, List<FunctionID> frames)
        {
            this.id = id;
            this.frames = frames;
            this.length = index;
        }
        public readonly int length;
        public readonly int id;
        public readonly List<FunctionID> frames;
    }
    public class FunctionInfo
    {
        public readonly string Namespace;
        public readonly string Signature;
        public FunctionInfo(string signature, string nameSpace)
        {
            this.Signature = signature;
            this.Namespace = nameSpace;
        }
    }
    public class Function
    {
        public Run run;
        public readonly bool reverse;
        public FunctionInfo Signature
        {
            get
            {
                return run.signatures[ID];
            }
        }
        public Function parent;
        public Function(FunctionID ID, Function parent, Run run, bool reverse)
        {
            if (parent == this)
            {
            }
            if (parent == null)
            {
            }
            this.parent = parent;
            this.run = run;
            this.ID = ID;
            this.reverse = reverse;
        }
        public readonly FunctionID ID;


        public int ExclusiveSamples
        {
            get
            {
                int samples = 0;
                foreach (StackWalk stackWalk in stackWalks)
                {
                    if (!reverse)
                    {
                        //if (stackWalk.length + 1 == stackWalk.frames.Count)
                        //{
                        //    samples++;
                        //}
                        if (stackWalk.length == 0)
                        {
                            samples++;
                        }
                    }
                    else
                    {
                        if (stackWalk.length + 1 == stackWalk.frames.Count)
                        {
                            samples++;
                        }
                    }

                    //if(stackWalk.frames[stackWalk.length-1]==this.ID)
                    //{
                    //    samples++;
                    //}
                }
                return samples;
            }
        }


        public int Samples;
        public int lastWalk;
        public double InclusiveChange;
        public double ExclusiveChange;
        private FunctionMap children;
        public List<StackWalk> stackWalks = new List<StackWalk>();
        public FunctionMap Children
        {
            get
            {
                if (children == null)
                {
                    children = new FunctionMap();
                    foreach (StackWalk walk in stackWalks)
                    {
                        if (walk.length != 0)
                        {
                            Function callee = Run.GetFunctionInfo(children, this, walk.frames[walk.length - 1], run, this.reverse);
                            if (callee.lastWalk != walk.id)
                            {
                                callee.Samples++;
                            }
                            callee.stackWalks.Add(new StackWalk(walk.id, walk.length - 1, walk.frames));
                        }
                    }
                }
                return children;
            }
        }
        public override string ToString()
        {
            return Signature.Signature;
        }
    }
    public class Run
    {
        private string executable;
        public static Function GetFunctionInfo(FunctionMap functions, Function parent, FunctionID id, Run run, bool reverse)
        {
            Function result;
            if (!functions.TryGetValue(id, out result))
            {
                result = new Function(id, parent, run, reverse);
                functions[id] = result;
            }
            return result;
        }
        public int maxSamples;
        public List<List<FunctionID>> stackWalks = new List<List<FunctionID>>();
        private void InterpreteData()
        {
            Interprete(functions, false, this);
            Interprete(callers, true, this);
            List<Function> startFunctions = new List<Function>();
            maxSamples = 0;
            foreach (List<FunctionID> stackWalk in stackWalks)
            {
                if (stackWalk.Count != 0)
                {
                    maxSamples++;
                }
            }
        }
        private void Interprete(FunctionMap map, bool reverse, Run run)
        {
            int currentWalk = 0;
            foreach (List<FunctionID> original in stackWalks)
            {
                currentWalk++;
                List<FunctionID> stackWalk;
                if (reverse)
                {
                    stackWalk = new List<FunctionID>(original);
                    stackWalk.Reverse();
                }
                else
                {
                    stackWalk = original;
                }
                //Function parent = null;
                for (int i = 0; i < stackWalk.Count; i++)
                {
                    Function function = Run.GetFunctionInfo(map, null, stackWalk[stackWalk.Count - i - 1], run, reverse);
                    //if (previous != null)
                    //{
                    //    previous.parent = function;
                    //} 
                    //previous = function;
                    //parent = function;
                    if (function.lastWalk != currentWalk)
                    {
                        function.Samples++;
                        function.stackWalks.Add(new StackWalk(currentWalk, stackWalk.Count - i - 1, stackWalk));
                    }
                    function.lastWalk = currentWalk;
                }
            }
            //foreach (List<FunctionID> stackWalk in stackWalks)
            //{
            //    Function f = Run.GetFunctionInfo(map, stackWalk[0], run,reverse);
            //    f.ExclusiveSamples++;
            //}
        }
        private string ReadString(BinaryReader br)
        {
            FunctionID length = br.ReadUInt64();
            byte[] abString = new byte[length];
            br.Read(abString, 0, (int)length);
            return System.Text.ASCIIEncoding.ASCII.GetString(abString, 0, (int)length);
        }
        private string FileName
        {
            get
            {
                return Path.Combine(Path.GetDirectoryName(Path.GetTempFileName()), Profiler.ProfilerGuid + ".nprof");
            }
        }
        public TimeSpan Duration
        {
            get
            {
                return this.end - this.start;
            }
        }
        private void ReadStackWalks()
        {
            signatures[0] = new FunctionInfo(Path.GetFileName(this.executable), "");
            using (BinaryReader r = new BinaryReader(File.Open(FileName, FileMode.Open)))
            {
                this.end = DateTime.Now;
                while (true)
                {
                    FunctionID functionId = r.ReadUInt64();
                    if (functionId == FunctionID.MaxValue)
                    {
                        break;
                    }
                    signatures[functionId] = new FunctionInfo(
                        ReadString(r),
                        ReadString(r)
                    );
                }
                while (true)
                {
                    List<FunctionID> stackWalk = new List<FunctionID>();
                    while (true)
                    {
                        FunctionID functionId = r.ReadUInt64();
                        if (functionId == FunctionID.MaxValue)
                        {
                            break;
                        }
                        else if (functionId == FunctionID.MaxValue - 1)
                        {
                            return;
                        }
                        stackWalk.Add(functionId);
                    }
                    stackWalk.Add(0);
                    stackWalks.Add(stackWalk);
                }
            }
        }
        public void Complete(object sender, EventArgs e)
        {
            if (File.Exists(FileName))
            {
                ReadStackWalks();
                File.Delete(FileName);
            }
            List<List<Function>> translated = new List<List<Function>>();
            FunctionMap map = new FunctionMap();
            //Run run = new Run(profiler, "");

            foreach (List<FunctionID> x in stackWalks)
            {
                List<Function> walk = new List<Function>();
                foreach (FunctionID id in x)
                {
                    Function info = Run.GetFunctionInfo(map, null, id, this, false);
                    walk.Add(info);
                }
                walk.Reverse();
                translated.Add(walk);

            }
            NProf.form.BeginInvoke(new EventHandler(delegate
            {
                InterpreteData();
                NProf.form.ShowRun(this);
            }));
        }
        public Dictionary<FunctionID, FunctionInfo> signatures = new Dictionary<FunctionID, FunctionInfo>();
        public FunctionMap functions = new FunctionMap();
        public FunctionMap callers = new FunctionMap();
        private DateTime start = DateTime.MinValue;
        private DateTime end = DateTime.MinValue;
        public Run(Profiler p, string executable)
        {
            this.profiler = p;
            this.executable = executable;
        }
        public string Executable
        {
            get
            {
                return this.executable;
            }
        }
        public bool Start(string fileName, string arguments)
        {
            start = DateTime.Now;
            return profiler.Start(this, fileName, arguments);
        }
        public Profiler profiler;
    }
    public class FunctionMap : Dictionary<FunctionID, Function>
    {
    }
    public class View : TreeView
    {
        public View()
        {
            this.Font = NProf.font;
            this.ShowPlusMinus = true;
        }
    }
    public class Profiler
    {
        public const string ProfilerGuid = "61D3E5D7-53E9-46B2-8DBC-011099A91793";
        public EventHandler completed;
        public bool Start(Run run, string fileName, string arguments)
        {
            this.run = run;
            process = new Process();
            //process.StartInfo = new ProcessStartInfo(application.Text);
            //process.StartInfo = new ProcessStartInfo(runView.application.Text);
            process.StartInfo = new ProcessStartInfo(fileName, arguments);
            process.StartInfo.EnvironmentVariables["COR_ENABLE_PROFILING"] = "0x1";
            process.StartInfo.EnvironmentVariables["COR_PROFILER"] = "{" + ProfilerGuid + "}";
            process.StartInfo.EnvironmentVariables["COMPLUS_ProfAPI_ProfilerCompatibilitySetting"] = "EnableV2Profiler";
            process.StartInfo.UseShellExecute = false;
            process.EnableRaisingEvents = true;
            process.Exited += new EventHandler(OnProcessExited);
            return process.Start();
        }
        private void OnProcessExited(object oSender, EventArgs ea)
        {
            completed(null, null);
        }
        private Run run;
        private Process process;
    }
}
