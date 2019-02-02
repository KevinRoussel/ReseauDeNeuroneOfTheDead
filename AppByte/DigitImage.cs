using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Transpilation
{
    public class DigitImage
    {
        byte _label;
        byte[][] _pixels;

        public int Label => (int)_label;
        public IEnumerable<byte> Pixels()
        {
            foreach(var i in _pixels )
                foreach(var j in i )
                    yield return j;
        }
        public IEnumerable<double> SampledPixels
            => Pixels().Select( i => ((double)i / 256)).ToList();

        public int Height => 28;
        public int Width => 28;

        public DigitImage( byte[][] pixels,
          byte label )
        {
            _pixels = new byte[28][];
            for( int i = 0; i < _pixels.Length; ++i )
                _pixels[i] = new byte[28];

            for( int i = 0; i < 28; ++i )
                for( int j = 0; j < 28; ++j )
                    _pixels[i][j] = pixels[i][j];

            _label = label;
        }

        public override string ToString()
        {
            string s = "";
            for( int i = 0; i < 28; ++i )
            {
                for( int j = 0; j < 28; ++j )
                {
                    if( _pixels[i][j] == 0 )
                        s += " "; // white
                    else if( _pixels[i][j] == 255 )
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += _label.ToString();
            return s;
        } // ToString

        public Tuple<string,int> ToFlatString()
        {
            StringBuilder sb = new StringBuilder();
            for( int i = 0; i < 28; ++i )
            {
                for( int j = 0; j < 28; ++j )
                {
                    sb.Append( _pixels[i][j] );
                    sb.Append( " " );
                }
                sb.Append( "\n" );
            }

            return new Tuple<string, int>( sb.ToString(), _label );
        }

    }
}
