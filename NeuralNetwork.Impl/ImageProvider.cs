using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace ITI.NeuralNetwork
{
    public class ImageProvider : IDisposable
    {
        FileStream ifsLabels;
        FileStream ifsImages;
        BinaryReader brLabels;
        BinaryReader brImages;

        byte[][] _pixels;
        List<DigitImage> _archive;
        bool _isFull;

        static string FindDataDirectory()
        {
            var start = AppDomain.CurrentDomain.BaseDirectory;
            string result = null;
            do
            {
                result = Directory.GetDirectories( start ).FirstOrDefault( i => i.Split( '\\' ).Last() == "Data" );
                start = Directory.GetParent( start ).FullName;

            } while( string.IsNullOrEmpty( result ) || Directory.GetParent( start ).FullName == start );
            return result;
        }

        public ImageProvider() : this( FindDataDirectory()+@"\train-labels.idx1-ubyte", FindDataDirectory()+@"\train-images.idx3-ubyte" ) { }
        
        public ImageProvider(string labelDatabasePath, string imagesDatabasePath)
        {
            // Opening data
            ifsLabels = new FileStream( labelDatabasePath, FileMode.Open, FileAccess.Read ); // test labels
            ifsImages = new FileStream( imagesDatabasePath, FileMode.Open, FileAccess.Read ); // test images
            brLabels = new BinaryReader( ifsLabels );
            brImages = new BinaryReader( ifsImages );

            // Consume meta data
            int magic1 = brImages.ReadInt32(); 
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();
            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            // Prepare image structure
            _pixels = new byte[28][];
            for( int i = 0; i < _pixels.Length; ++i )
                _pixels[i] = new byte[28];

            // Prepare image cache
            _archive = new List<DigitImage>();
            _isFull = false;
        }

        public void Dispose()
        {
            ifsImages.Dispose();
            ifsImages.Dispose();
            brLabels.Dispose();
            brImages.Dispose();
        }

        public IEnumerable<DigitImage> ImageStream()
        {
            if( _isFull ) yield break;

            for( int di = 0; di < 10000; ++di )
            {
                for( int i = 0; i < 28; ++i )
                {
                    for( int j = 0; j < 28; ++j )
                    {
                        byte b = brImages.ReadByte();
                        _pixels[i][j] = b;
                    }
                }

                byte lbl = brLabels.ReadByte();
                DigitImage dImage = new DigitImage( _pixels, lbl );
                _archive.Add( dImage );
                yield return dImage;
            }
            _isFull = true;
            yield break;
        }

    }
}
