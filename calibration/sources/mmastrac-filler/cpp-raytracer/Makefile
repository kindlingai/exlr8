#!gnumake

TARGET = rt
CXXFILES = SceneObject.cpp linearalgebra.cpp Intersection.cpp RayTracer.cpp main.cpp Sphere.cpp Square.cpp SceneObjectList.cpp Plane.cpp FileReader.cpp Light.cpp Box.cpp Triangle.cpp
CFILES = 
METACXXFILES = 
QTDIR = 

-include Makefile.dirs
-include Makefile.defs

MOC = moc
CXXFLAGS = -O6
CFLAGS =  
LDDIRS = -L/usr/X11R6/lib
QTLIBS = 
LDLIBS = -lm
LDFLAGS = $(LDOPT) $(LDDIRS) $(LDLIBS) 

CXXDEPEND = $(CXX) -M
CDEPEND = $(CC) -M

SOURCE_OBJECTS = $(CXXFILES:.cpp=.o) 
SOURCE_OBJECTS += $(CFILES:.c=.o)
META_OBJECTS = $(METACXXFILES:.cpp=.o)
OBJECTS = $(sort $(SOURCE_OBJECTS))
OBJECTS += $(sort $(META_OBJECTS))

.SUFFIXES: .c++ .cpp

.cpp.o: 
	$(CXX) -c $(CXXFLAGS) $<

.c++.o: 
	$(CXX) -c $(CXXFLAGS) $<

.c.o: 
	$(CC) -c $(CFLAGS) $<

moc/%.cpp: %.h
	test -d moc || mkdir moc
	$(MOC) $< -o $@

moc/%.o: moc/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(TARGET): $(OBJECTS) 
	$(CXX) -g $(OBJECTS) -o $(TARGET) $(LDFLAGS)

clean : 
	rm -f .Makedepend ; rm -f *.o ; rm -f moc/* ; rm -f core ; rm -f *~

relink: rmtarget $(TARGET)

rmtarget:
	rm -f $(TARGET)

rebuild:
	make clean
	make

depend:
	$(RM) .Makedepend
	touch .Makedepend
ifdef CXXFILES
	$(CXXDEPEND) $(CXXFLAGS) $(CXXFILES) >> .Makedepend
endif
ifdef CFILES
	$(CDEPEND) $(CFLAGS) $(CFILES) >> .Makedepend
endif
	rm -f /tmp/.Makedepend
	sed -e 's:.c++.o:.o:' .Makedepend > /tmp/.Makedepend
	mv /tmp/.Makedepend .

-include .Makedepend







