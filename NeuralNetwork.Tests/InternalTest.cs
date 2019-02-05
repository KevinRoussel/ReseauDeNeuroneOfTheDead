using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using NUnit.Framework;

using ITI.NeuralNetwork;

namespace ITI.NeuralNetwork.Tests
{
    [TestFixture]
    public class InternalTest
    {
        [Test]
        public void Test1()
        {
            Assert.True(true);
        }

        [Test]
        public void ValidateDatabaseInput()
        {

            using( ImageProvider i = new ImageProvider() )
            {
                var stream = i.ImageStream().GetEnumerator();

                stream.MoveNext();
                Assert.That(stream.Current.Label, Is.EqualTo(5) );

                stream.MoveNext();
                Assert.That( stream.Current.Label, Is.EqualTo( 0 ) );

                stream.MoveNext();
                Assert.That( stream.Current.Label, Is.EqualTo( 4 ) );

                stream.MoveNext();
                Assert.That( stream.Current.Label, Is.EqualTo( 1 ) );

                stream.MoveNext();
                Assert.That( stream.Current.Label, Is.EqualTo( 9 ) );
            }
        }

        [Test]
        public void InitialFileClamped0to1()
        {
            using( ImageProvider i = new ImageProvider() )
            {
                foreach( var el in i.ImageStream() )
                {
                    foreach(var pixel in el.SampledPixels)
                    {
                        Assert.That( pixel, Is.InRange(0, 1) );
                    }
                }
            }
        }
        [Test]
        public void InitialFileClamped0to256()
        {
            using( ImageProvider i = new ImageProvider() )
            {
                foreach( var el in i.ImageStream() )
                {
                    foreach(var pixel in el.Pixels())
                    {
                        Assert.That( pixel, Is.InRange(0, 255 ));
                    }
                }
            }
        }


    }
}
